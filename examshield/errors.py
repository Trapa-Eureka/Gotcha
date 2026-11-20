class ExamShieldError(Exception):
    exit_code = 2


class NetworkRefusedError(ExamShieldError):
    pass


class ModelUnavailableError(ExamShieldError):
    pass


class ModelAbortError(ExamShieldError):
    """이 모델로는 더 호출해도 소용없는 실패. 키가 틀렸거나 모델 이름이 없는 경우다."""
