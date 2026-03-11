class ModelNotLoadedError(Exception):
    pass


class ModerationTaskNotFoundError(Exception):
    pass


class ItemNotFoundError(Exception):
    pass


class AuthError(Exception):
    pass


class InvalidCredentialsError(AuthError):
    pass


class BlockedAccountError(AuthError):
    pass


class InvalidTokenError(AuthError):
    pass
