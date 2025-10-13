# 기존 이름 유지: 내부는 tuc.cli build 호출만
from tuc.cli import main as _main
import sys
sys.argv = ["", "build"]
_main()
