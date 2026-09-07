#!/usr/bin/env node

import { createServer } from "node:http";
import { spawn } from "node:child_process";
import { createInterface } from "node:readline";
import { chmod, readFile, rename, writeFile } from "node:fs/promises";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StreamableHTTPClientTransport } from "@modelcontextprotocol/sdk/client/streamableHttp.js";
import { UnauthorizedError } from "@modelcontextprotocol/sdk/client/auth.js";

const DEFAULT_ENDPOINT = "https://mcp.deepl.com/v1/mcp";
const DEFAULT_CALLBACK_PORT = 8765;
const OAUTH_ENV_NAME = "DEEPL_OAUTH_CREDENTIALS";

function log(message) {
  process.stderr.write(`[deepl-mcp] ${message}\n`);
}

function errorMessage(error) {
  return error instanceof Error ? `${error.name}: ${error.message}` : String(error);
}

function encodeOAuthData(data) {
  return Buffer.from(JSON.stringify(data), "utf8").toString("base64url");
}

function decodeOAuthData(value) {
  const parsed = JSON.parse(Buffer.from(value, "base64url").toString("utf8"));
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new Error("OAuth credentials are not a JSON object");
  }
  return parsed;
}

async function writeEnvCredential(envFile, value) {
  if (!envFile) throw new Error("DeepL OAuth .env path is missing");
  let source = "";
  try {
    source = await readFile(envFile, "utf8");
  } catch (error) {
    if (error?.code !== "ENOENT") throw error;
  }
  const newline = source.includes("\r\n") ? "\r\n" : "\n";
  const replacement = `${OAUTH_ENV_NAME}=${value}`;
  const lines = source ? source.replace(/\r?\n$/, "").split(/\r?\n/) : [];
  let replaced = false;
  const updated = [];
  for (const line of lines) {
    if (new RegExp(`^\\s*(?:export\\s+)?${OAUTH_ENV_NAME}\\s*=`).test(line)) {
      if (!replaced) updated.push(replacement);
      replaced = true;
    } else {
      updated.push(line);
    }
  }
  if (!replaced) updated.push(replacement);
  const temporary = `${envFile}.${process.pid}.tmp`;
  await writeFile(temporary, `${updated.join(newline)}${newline}`, {
    encoding: "utf8",
    mode: 0o600,
  });
  await rename(temporary, envFile);
  await chmod(envFile, 0o600).catch(() => {});
}

class EnvFileOAuthProvider {
  constructor({ redirectUrl, envFile, onRedirect }) {
    this._redirectUrl = redirectUrl;
    this._envFile = envFile;
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
    const encoded = process.env[OAUTH_ENV_NAME];
    if (!encoded) return;
    try {
      this._data = decodeOAuthData(encoded);
    } catch (error) {
      throw new Error(
        `Cannot read DeepL OAuth credentials from ${OAUTH_ENV_NAME}: ${errorMessage(error)}`,
      );
    }
  }

  async _save() {
    try {
      const encoded = encodeOAuthData(this._data);
      await writeEnvCredential(this._envFile, encoded);
      process.env[OAUTH_ENV_NAME] = encoded;
    } catch (error) {
      throw new Error(
        `Cannot save DeepL OAuth credentials to .env: ${errorMessage(error)}`,
      );
    }
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
    log("OAuth credentials updated in .env.");
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
  const address = url.toString();
  let command;
  let args;
  if (process.platform === "darwin") {
    command = "/usr/bin/open";
    args = [address];
  } else if (process.platform === "win32") {
    command = "rundll32.exe";
    args = ["url.dll,FileProtocolHandler", address];
  } else if (process.platform === "linux") {
    command = "xdg-open";
    args = [address];
  } else {
    log(`Cannot open a browser automatically on ${process.platform}.`);
    log(`Open this URL manually: ${address}`);
    return Promise.resolve();
  }
  return new Promise((resolve) => {
    const child = spawn(command, args, {
      stdio: "ignore",
      detached: true,
    });
    child.once("error", (error) => {
      log(`Cannot open browser automatically: ${errorMessage(error)}`);
      log(`Open this URL manually: ${address}`);
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
    this.envFile = options.envFile;
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
    const provider = new EnvFileOAuthProvider({
      redirectUrl,
      envFile: this.envFile,
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
        log("OAuth callback received; exchanging the authorization code.");
        await firstTransport.finishAuth(callback.code);
        log("OAuth authorization code exchange completed.");
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

export { decodeOAuthData, encodeOAuthData, writeEnvCredential };

let activeClient = null;

async function handleRequest(request) {
  if (!request || typeof request !== "object") throw new Error("Request must be a JSON object");
  const options = {
    endpoint: request.endpoint,
    callbackPort: request.oauth_callback_port,
    envFile: request.env_file,
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
