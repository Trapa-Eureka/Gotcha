"""단일 실행 파일의 시작점."""

import multiprocessing

from examshield.cli import app

if __name__ == "__main__":
    # protect --copies가 띄우는 작업 프로세스는 이 실행 파일을 다시 실행한다. 이 줄이 없으면
    # 작업 프로세스가 명령줄을 처음부터 다시 돌린다.
    multiprocessing.freeze_support()
    app()
