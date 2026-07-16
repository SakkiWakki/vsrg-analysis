"""Game-agnostic fullscreen shader system.

- stack.py       pure sampling: shader events -> per-frame passes
- library/       builtin GLSL effects sharing one uniform contract
- gl_pipeline.py Qt/GL execution (the only GL-specific piece)

`ShaderGLPipeline` is imported from its module directly by GL hosts;
importing it here would pull QtOpenGL into headless users of the
sampling side.
"""
from analysis.player.render.shaders.stack import ShaderStackEffect

__all__ = ['ShaderStackEffect']
