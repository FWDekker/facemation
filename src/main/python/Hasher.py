import base64
import hashlib


def _tob64(digest: bytes) -> str:
    """
    Encodes [digest] to a "!@"-base64-string.

    :param digest: a hash digest
    :return: [digest] as a "!@"-base64-string
    """

    return base64.b64encode(digest, altchars=b"!@").decode()


def hash_file(filename: str) -> str:
    """
    Calculates the "!@"-base64-encoded SHA3 hash of the contents of [filename].

    Taken from https://stackoverflow.com/a/44873382.

    :param filename: the path to the file to calculate the hash of
    :return: the SHA3 hash of the contents of [filename]
    """

    h = hashlib.shake_128()

    b = bytearray(128 * 1024)
    mv = memoryview(b)
    with open(filename, "rb", buffering=0) as file:
        # noinspection PyUnresolvedReferences
        while n := file.readinto(mv):
            h.update(mv[:n])

    return _tob64(h.digest(64))


def hash_string(string: str) -> str:
    """
    Calculates the "!@"-base64-encoded SHA3 hash of [string].

    :param string: the string to calculate the hash of
    :return: the SHA3 hash of [string]
    """

    return _tob64(hashlib.shake_128(string.encode("utf-8")).digest(64))
