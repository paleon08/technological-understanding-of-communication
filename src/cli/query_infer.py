# 기존처럼 --query / --k 지원
from tuc.cli import main as _main
import sys
argv = ["", "query"]
# 전달 인자 유지
argv += [arg for arg in sys.argv[1:]]
sys.argv = argv
_main()
