import hashlib


def sha256sum(filename: str) -> str:
    """
    Calculates the sha256 hash of the contents of [filename].

    Taken from https://stackoverflow.com/a/44873382.

    :param filename: the path to the file to calculate the hash of
    :return: the sha256 hash of the contents of [filename]
    """

    h = hashlib.sha256()
    b = bytearray(128 * 1024)
    mv = memoryview(b)
    with open(filename, "rb", buffering=0) as file:
        # noinspection PyUnresolvedReferences
        while n := file.readinto(mv):
            h.update(mv[:n])
    return h.hexdigest()


def sha256sums(string: str) -> str:
    """
    Calculates the sha256 hash of [string].

    :param string: the string to calculate the hash of
    :return: the sha256 hash of [string]
    """

    return hashlib.sha256(string.encode("utf-8")).hexdigest()
