"""Colab + local 환경 setup. token assert + git credential helper.

노트북 첫 셀에서 (sys.path 등록 + drive.mount 후):
    from src.colab_setup import setup
    setup()  # Colab 시 GITHUB_TOKEN assert + ~/.git-credentials + global config. local 은 no-op.

⚠ sys.path 등록 / cwd 변경 / drive.mount 는 *이 모듈 import 전* 노트북 inline 으로 해야 함
   (chicken-egg: 모듈 import 자체가 sys.path 의존). 이 모듈은 sys.path 가 잡힌 후 호출 가정.

⚠ utils.load_secrets() 는 src.utils import 시 자동 호출 — .env / Colab userdata 의 HF_TOKEN /
   OPENAI_API_KEY / GITHUB_TOKEN / GIT_USER_NAME / GIT_USER_EMAIL 등 자동 환경변수 set.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def is_colab() -> bool:
    """Colab 환경인지."""
    try:
        import google.colab  # type: ignore  # noqa: F401
        return True
    except ImportError:
        return False


def assert_github_token() -> None:
    """Colab 환경에서 GITHUB_TOKEN userdata 강제 검증 + 환경변수 set.

    local 에선 no-op (utils.load_secrets 가 .env 의 GITHUB_TOKEN 자동 로드, 없어도 silent).
    use case: private repo clone / GitHub API / `!git pull` 등 GITHUB_TOKEN 이 *반드시* 필요한 경우.
    """
    if not is_colab():
        return
    from google.colab import userdata  # type: ignore
    token = userdata.get('GITHUB_TOKEN')
    assert token is not None, "Colab Secrets에 GITHUB_TOKEN이 없거나 Notebook access가 꺼져 있습니다."
    os.environ['GITHUB_TOKEN'] = token


def setup_git_credential_helper() -> None:
    """GITHUB_TOKEN 을 git CLI 가 자동 사용하도록 ~/.git-credentials 박아두기.

    askpass 방식 대신 ~/.git-credentials + `credential.helper store` 사용 —
    노트북 안에서 `!git pull` 같은 subprocess 호출 (interactive prompt 못 띄움) 에서 더 robust.

    Colab 에서만 동작 — local 은 보통 ~/.gitconfig + credential helper 이미 setup.
    GITHUB_TOKEN 환경변수 없으면 silent skip.

    GIT_USER_NAME / GIT_USER_EMAIL 환경변수가 있으면 git config user.* 도 같이 set
    (Colab 은 매 세션 ~/.gitconfig 가 비어있어 매번 set 필요).
    """
    if not is_colab():
        return
    token = os.getenv('GITHUB_TOKEN')
    if not token:
        return  # token 없으면 skip — assert_github_token 이 먼저 강제하면 보통 있음

    # 1) ~/.git-credentials 에 token 박기 (URL 별 자동 사용). perm 600 으로 다른 user read 차단.
    cred_path = Path.home() / '.git-credentials'
    cred_path.write_text(f'https://x-access-token:{token}@github.com\n')
    cred_path.chmod(0o600)

    # 2) git global config — credential.helper store + (있으면) user.name/email
    subprocess.run(['git', 'config', '--global', 'credential.helper', 'store'], check=False)
    if name := os.getenv('GIT_USER_NAME'):
        subprocess.run(['git', 'config', '--global', 'user.name', name], check=False)
    if email := os.getenv('GIT_USER_EMAIL'):
        subprocess.run(['git', 'config', '--global', 'user.email', email], check=False)


def setup(*, require_github_token: bool = True, configure_git: bool = True) -> None:
    """Colab + local 환경 setup 한 번에.

    - Colab + require_github_token=True: GITHUB_TOKEN userdata assert.
    - Colab + configure_git=True: ~/.git-credentials + global config (GIT_USER_NAME/EMAIL 있으면 set).
    - local: 둘 다 no-op (utils.load_secrets 가 .env 자동 로드 + ~/.gitconfig 가정).

    HF_TOKEN / OPENAI_API_KEY / GITHUB_TOKEN / GIT_USER_NAME / GIT_USER_EMAIL 의 .env 로드는
    utils.load_secrets() 가 자동 처리 (src.utils import 시점).

    drive.mount / sys.path / cwd 는 노트북 첫 셀에서 inline 처리 (chicken-egg 회피).
    """
    if require_github_token:
        assert_github_token()
    if configure_git:
        setup_git_credential_helper()
