#include <CoreFoundation/CoreFoundation.h>
#include <Security/Security.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fail(const char *message, int code) {
    fprintf(stderr, "%s\n", message);
    return code;
}

static CFStringRef make_string(const char *value) {
    return CFStringCreateWithCString(
        kCFAllocatorDefault, value, kCFStringEncodingUTF8
    );
}

static CFMutableDictionaryRef make_query(
    CFStringRef service, CFStringRef account
) {
    CFMutableDictionaryRef query = CFDictionaryCreateMutable(
        kCFAllocatorDefault,
        0,
        &kCFTypeDictionaryKeyCallBacks,
        &kCFTypeDictionaryValueCallBacks
    );
    CFDictionarySetValue(query, kSecClass, kSecClassGenericPassword);
    CFDictionarySetValue(query, kSecAttrService, service);
    CFDictionarySetValue(query, kSecAttrAccount, account);
    return query;
}

static CFDataRef read_stdin(void) {
    size_t capacity = 4096;
    size_t length = 0;
    unsigned char *buffer = malloc(capacity);
    if (buffer == NULL) return NULL;
    while (!feof(stdin)) {
        if (length == capacity) {
            capacity *= 2;
            unsigned char *expanded = realloc(buffer, capacity);
            if (expanded == NULL) {
                free(buffer);
                return NULL;
            }
            buffer = expanded;
        }
        length += fread(buffer + length, 1, capacity - length, stdin);
        if (ferror(stdin)) {
            free(buffer);
            return NULL;
        }
    }
    if (length == 0) {
        free(buffer);
        return NULL;
    }
    CFDataRef data = CFDataCreate(kCFAllocatorDefault, buffer, (CFIndex)length);
    free(buffer);
    return data;
}

int main(int argc, const char *argv[]) {
    if (argc != 4) {
        return fail("usage: deepl-keychain-helper get|set|delete service account", 2);
    }
    CFStringRef service = make_string(argv[2]);
    CFStringRef account = make_string(argv[3]);
    if (service == NULL || account == NULL) {
        return fail("invalid UTF-8 service or account", 2);
    }
    CFMutableDictionaryRef query = make_query(service, account);
    OSStatus status = errSecSuccess;

    if (strcmp(argv[1], "get") == 0) {
        CFDictionarySetValue(query, kSecReturnData, kCFBooleanTrue);
        CFDictionarySetValue(query, kSecMatchLimit, kSecMatchLimitOne);
        CFTypeRef result = NULL;
        status = SecItemCopyMatching(query, &result);
        if (status == errSecItemNotFound) {
            CFRelease(query);
            CFRelease(service);
            CFRelease(account);
            return 44;
        }
        if (status != errSecSuccess || result == NULL ||
            CFGetTypeID(result) != CFDataGetTypeID()) {
            if (result != NULL) CFRelease(result);
            return fail("Keychain read failed", 1);
        }
        CFDataRef data = (CFDataRef)result;
        fwrite(
            CFDataGetBytePtr(data),
            1,
            (size_t)CFDataGetLength(data),
            stdout
        );
        CFRelease(result);
    } else if (strcmp(argv[1], "set") == 0) {
        CFDataRef data = read_stdin();
        if (data == NULL) return fail("refusing to store empty Keychain data", 2);
        const void *keys[] = {kSecValueData};
        const void *values[] = {data};
        CFDictionaryRef changes = CFDictionaryCreate(
            kCFAllocatorDefault,
            keys,
            values,
            1,
            &kCFTypeDictionaryKeyCallBacks,
            &kCFTypeDictionaryValueCallBacks
        );
        status = SecItemUpdate(query, changes);
        CFRelease(changes);
        if (status == errSecItemNotFound) {
            CFDictionarySetValue(query, kSecValueData, data);
            status = SecItemAdd(query, NULL);
        }
        CFRelease(data);
        if (status != errSecSuccess) return fail("Keychain write failed", 1);
    } else if (strcmp(argv[1], "delete") == 0) {
        status = SecItemDelete(query);
        if (status != errSecSuccess && status != errSecItemNotFound) {
            return fail("Keychain delete failed", 1);
        }
    } else {
        return fail("unknown command", 2);
    }

    CFRelease(query);
    CFRelease(service);
    CFRelease(account);
    return 0;
}
