import configargparse
from config.options import BaseOptions  # 按你的实际路径改

def add_matcher_args(parser: configargparse.ArgumentParser):
    # ---- dataset / GT ----
    parser.add_argument("--opt_inliers_only", action="store_true")
    parser.add_argument("--joint_train", type=bool, default= False)
    parser.add_argument("--freeze_detector", type=bool, default= True)
    # parser.set_defaults(num_workers=0)
    return parser


class MatcherOptions(BaseOptions):
    """
    继承 BaseOptions，把 matcher 的 args 合并进去。
    用法：from matcher_options import MatcherOptions
         opt = MatcherOptions().parse()
    """
    def initialize(self, parser: configargparse.ArgumentParser):
        # 1) 先加载基础参数
        parser = super().initialize(parser)

        # 2) 再加载 matcher 参数（新增）
        parser = add_matcher_args(parser)

        return parser