#!/usr/bin/env node

import { createServer } from "node:http";
import { spawn } from "node:child_process";
import { createInterface } from "node:readline";
import { mkdir, stat } from "node:fs/promises";
import { homedir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StreamableHTTPClientTransport } from "@modelcontextprotocol/sdk/client/streamableHttp.js";
import { UnauthorizedError } from "@modelcontextprotocol/sdk/client/auth.js";

const DEFAULT_ENDPOINT = "https://mcp.deepl.com/v1/mcp";
const DEFAULT_CALLBACK_PORT = 8765;
const DEFAULT_KEYCHAIN_SERVICE = "Beast Academy OCR DeepL MCP";

function log(message) {
  process.stderr.write(`[deepl-mcp] ${message}\n`);
}

function errorMessage(error) {
  return error instanceof Error ? `${error.name}: ${error.message}` : String(error);
}

function runProcess(command, args, input = null) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      stdio: ["pipe", "pipe", "pipe"],
    });
    const stdout = [];
    const stderr = [];
    child.stdout.on("data", (chunk) => {
      stdout.push(chunk);
    });
    child.stderr.on("data", (chunk) => {
      stderr.push(chunk);
    });
    child.on("error", reject);
    child.on("close", (code) => {
      if (code === 0) {
        resolve(Buffer.concat(stdout));
      } else {
        const detail = Buffer.concat(stderr).toString("utf8").trim();
        const error = new Error(detail || `${command} exited with ${code}`);
        error.exitCode = code;
        reject(error);
      }
    });
    if (input === null) {
      child.stdin.end();
    } else {
      child.stdin.end(input);
    }
  });
}

let keychainHelperPromise;

async function keychainHelperPath() {
  if (keychainHelperPromise) return keychainHelperPromise;
  keychainHelperPromise = (async () => {
    const source = fileURLToPath(new URL("./deepl_keychain_helper.c", import.meta.url));
    const cacheDirectory = join(homedir(), "Library", "Caches", "BeastAcademyOCR");
    const binary = join(cacheDirectory, "deepl-keychain-helper");
    await mkdir(cacheDirectory, { recursive: true, mode: 0o700 });
    let rebuild = false;
    try {
      const [sourceInfo, binaryInfo] = await Promise.all([stat(source), stat(binary)]);
      rebuild = sourceInfo.mtimeMs > binaryInfo.mtimeMs;
    } catch {
      rebuild = true;
    }
    if (rebuild) {
      log("Building the local macOS Keychain helper.");
      await runProcess("/usr/bin/clang", [
        source,
        "-framework",
        "Security",
        "-framework",
        "CoreFoundation",
        "-o",
        binary,
      ]);
    }
    return binary;
  })();
  return keychainHelperPromise;
}

async function runKeychain(command, service, account, input = null) {
  const helper = await keychainHelperPath();
  return runProcess(helper, [command, service, account], input);
}

class KeychainOAuthProvider {
  constructor({ redirectUrl, endpoint, service, account, onRedirect }) {
    this._redirectUrl = redirectUrl;
    this._endpoint = endpoint;
    this._service = service;
    this._account = account;
    this._onRedirect = onRedirect;
    this._loaded = false;
    this._data = {};
  }

  get redirectUrl() {
    return this._redirectUrl;
  }

  get clientMetadata() {
    return {
      client_name: "Beast Academy OCR DeepL MCP",
      redirect_uris: [this._redirectUrl],
      grant_types: ["authorization_code", "refresh_token"],
      response_types: ["code"],
      token_endpoint_auth_method: "none",
    };
  }

  async _load() {
    if (this._loaded) return;
    this._loaded = true;
    try {
      const raw = await runKeychain("get", this._service, this._account);
      const parsed = JSON.parse(raw.toString("utf8"));
      if (parsed && typeof parsed === "object") this._data = parsed;
    } catch (error) {
      if (error.exitCode !== 44) {
        throw new Error(`Cannot read OAuth data from Keychain: ${errorMessage(error)}`);
      }
    }
  }

  async _save() {
    await runKeychain(
      "set",
      this._service,
      this._account,
      JSON.stringify(this._data),
    );
  }

  async clientInformation() {
    await this._load();
    return this._data.clientInformation;
  }

  async saveClientInformation(clientInformation) {
    await this._load();
    this._data.clientInformation = clientInformation;
    await this._save();
  }

  async tokens() {
    await this._load();
    return this._data.tokens;
  }

  async saveTokens(tokens) {
    await this._load();
    this._data.tokens = tokens;
    await this._save();
  }

  async redirectToAuthorization(authorizationUrl) {
    await this._onRedirect(authorizationUrl);
  }

  async saveCodeVerifier(codeVerifier) {
    await this._load();
    this._data.codeVerifier = codeVerifier;
    await this._save();
  }

  async codeVerifier() {
    await this._load();
    if (!this._data.codeVerifier) throw new Error("OAuth PKCE verifier is unavailable");
    return this._data.codeVerifier;
  }

  async saveDiscoveryState(discoveryState) {
    await this._load();
    this._data.discoveryState = discoveryState;
    await this._save();
  }

  async discoveryState() {
    await this._load();
    return this._data.discoveryState;
  }

  async invalidateCredentials(scope) {
    await this._load();
    if (scope === "all" || scope === "client") delete this._data.clientInformation;
    if (scope === "all" || scope === "tokens") delete this._data.tokens;
    if (scope === "all" || scope === "verifier") delete this._data.codeVerifier;
    if (scope === "all" || scope === "discovery") delete this._data.discoveryState;
    await this._save();
  }
}

function startCallbackServer(port) {
  let settle;
  let fail;
  const callback = new Promise((resolve, reject) => {
    settle = resolve;
    fail = reject;
  });
  const server = createServer((request, response) => {
    const url = new URL(request.url || "/", `http://127.0.0.1:${port}`);
    if (url.pathname !== "/callback") {
      response.writeHead(404, { "content-type": "text/plain; charset=utf-8" });
      response.end("Not found");
      return;
    }
    const code = url.searchParams.get("code");
    const oauthError = url.searchParams.get("error");
    if (code) {
      response.writeHead(200, { "content-type": "text/html; charset=utf-8" });
      response.end(
        "<!doctype html><meta charset=utf-8><title>DeepL authorization complete</title>" +
          "<h1>DeepL 授权成功</h1><p>可以关闭此窗口并返回终端。</p>",
      );
      settle({ code, state: url.searchParams.get("state") });
      return;
    }
    const message = oauthError || "OAuth callback did not contain an authorization code";
    response.writeHead(400, { "content-type": "text/plain; charset=utf-8" });
    response.end(message);
    fail(new Error(message));
  });
  const listening = new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(port, "127.0.0.1", () => resolve());
  });
  return {
    callback,
    listening,
    close: () => new Promise((resolve) => server.close(() => resolve())),
  };
}

function openBrowser(url) {
  return new Promise((resolve) => {
    const child = spawn("/usr/bin/open", [url.toString()], {
      stdio: "ignore",
      detached: true,
    });
    child.once("error", (error) => {
      log(`Cannot open browser automatically: ${errorMessage(error)}`);
      log(`Open this URL manually: ${url.toString()}`);
      resolve();
    });
    child.once("spawn", () => {
      child.unref();
      resolve();
    });
  });
}

class DeepLMcpClient {
  constructor(options) {
    this.endpoint = options.endpoint || DEFAULT_ENDPOINT;
    this.callbackPort = Number(options.callbackPort || DEFAULT_CALLBACK_PORT);
    this.keychainService = options.keychainService || DEFAULT_KEYCHAIN_SERVICE;
    this.keychainAccount = options.keychainAccount || this.endpoint;
    this.client = null;
    this.transport = null;
  }

  async _newConnection(provider) {
    const transport = new StreamableHTTPClientTransport(new URL(this.endpoint), {
      authProvider: provider,
    });
    const client = new Client(
      { name: "beast-academy-ocr-deepl", version: "0.1.0" },
      { capabilities: {} },
    );
    await client.connect(transport);
    return { client, transport };
  }

  async connect() {
    if (this.client) return;
    const callbackServer = startCallbackServer(this.callbackPort);
    try {
      await callbackServer.listening;
    } catch (error) {
      throw new Error(
        `Cannot listen on OAuth callback port ${this.callbackPort}: ${errorMessage(error)}`,
      );
    }
    const redirectUrl = `http://127.0.0.1:${this.callbackPort}/callback`;
    const provider = new KeychainOAuthProvider({
      redirectUrl,
      endpoint: this.endpoint,
      service: this.keychainService,
      account: this.keychainAccount,
      onRedirect: async (authorizationUrl) => {
        log("Opening the DeepL OAuth authorization page in the browser.");
        log("The project never receives or stores your DeepL password.");
        await openBrowser(authorizationUrl);
      },
    });

    try {
      const firstTransport = new StreamableHTTPClientTransport(new URL(this.endpoint), {
        authProvider: provider,
      });
      const firstClient = new Client(
        { name: "beast-academy-ocr-deepl", version: "0.1.0" },
        { capabilities: {} },
      );
      try {
        await firstClient.connect(firstTransport);
        this.client = firstClient;
        this.transport = firstTransport;
      } catch (error) {
        if (!(error instanceof UnauthorizedError)) {
          await firstTransport.close().catch(() => {});
          throw error;
        }
        let timeoutId;
        const timeout = new Promise((_, reject) => {
          timeoutId = setTimeout(
            () => reject(new Error("Timed out waiting for DeepL OAuth authorization")),
            300000,
          );
        });
        let callback;
        try {
          callback = await Promise.race([callbackServer.callback, timeout]);
        } finally {
          clearTimeout(timeoutId);
        }
        await firstTransport.finishAuth(callback.code);
        await firstTransport.close().catch(() => {});
        const connected = await this._newConnection(provider);
        this.client = connected.client;
        this.transport = connected.transport;
      }
    } finally {
      await callbackServer.close().catch(() => {});
    }
    const tools = await this.client.listTools();
    if (!tools.tools.some((tool) => tool.name === "translate-text")) {
      throw new Error("DeepL MCP does not expose translate-text");
    }
    log(`Connected to ${this.endpoint}; translate-text is available.`);
  }

  async translate(request) {
    await this.connect();
    const args = {
      text: request.text,
      targetLang: request.target_lang,
    };
    if (request.source_lang) args.sourceLang = request.source_lang;
    if (request.formality) args.formality = request.formality;
    if (request.glossary_id) args.glossaryId = request.glossary_id;
    if (request.style_id) args.styleId = request.style_id;
    if (request.context) args.context = request.context;
    if (Array.isArray(request.custom_instructions) && request.custom_instructions.length) {
      args.customInstructions = request.custom_instructions;
    }
    const result = await this.client.callTool({
      name: "translate-text",
      arguments: args,
    });
    if (result.isError) {
      const detail = result.content?.map((item) => item.text || "").join(" ") || "unknown error";
      throw new Error(`DeepL translate-text failed: ${detail}`);
    }
    for (const item of result.content || []) {
      if (item.type !== "text" || !item.text) continue;
      try {
        const parsed = JSON.parse(item.text);
        if (parsed && typeof parsed.text === "string") return parsed;
      } catch {
        return { text: item.text };
      }
    }
    if (result.structuredContent && typeof result.structuredContent.text === "string") {
      return result.structuredContent;
    }
    throw new Error("DeepL returned no translated text");
  }

  async health() {
    await this.connect();
    return { endpoint: this.endpoint, tool: "translate-text" };
  }

  async close() {
    if (this.transport) await this.transport.close().catch(() => {});
    this.transport = null;
    this.client = null;
  }
}

let activeClient = null;

async function handleRequest(request) {
  if (!request || typeof request !== "object") throw new Error("Request must be a JSON object");
  const options = {
    endpoint: request.endpoint,
    callbackPort: request.oauth_callback_port,
    keychainService: request.oauth_keychain_service,
    keychainAccount: request.oauth_keychain_account,
  };
  if (!activeClient) activeClient = new DeepLMcpClient(options);
  if (request.action === "health") return activeClient.health();
  if (request.action === "translate") {
    if (typeof request.text !== "string" || !request.text.trim()) {
      throw new Error("translate requires non-empty text");
    }
    if (typeof request.target_lang !== "string" || !request.target_lang.trim()) {
      throw new Error("translate requires target_lang");
    }
    return activeClient.translate(request);
  }
  throw new Error(`Unknown action: ${request.action}`);
}

const lines = createInterface({ input: process.stdin, crlfDelay: Infinity });
for await (const line of lines) {
  if (!line.trim()) continue;
  let request;
  try {
    request = JSON.parse(line);
    const result = await handleRequest(request);
    process.stdout.write(`${JSON.stringify({ id: request.id, ok: true, result })}\n`);
  } catch (error) {
    process.stdout.write(
      `${JSON.stringify({ id: request?.id ?? null, ok: false, error: errorMessage(error) })}\n`,
    );
  }
}

if (activeClient) await activeClient.close();
