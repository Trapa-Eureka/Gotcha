class ExamShieldError(Exception):
    exit_code = 2


class NetworkRefusedError(ExamShieldError):
    pass


class ModelUnavailableError(ExamShieldError):
    pass
