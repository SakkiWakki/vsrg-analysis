// Windows build wrapper for render.c.
//
// Pulls in win_gl_loader.h (which provides <windows.h> + <GL/gl.h> and
// the extension pointer redirects) then fakes out the Linux GL include
// guards so render.c's "#include <GL/gl.h>" and "#include <GL/glext.h>"
// become no-ops. render.c and nanovg_gl.h then compile against our
// hand-rolled loader with no other changes.

#include "win_gl_loader.h"

// Suppress the Linux GL headers that render.c would pull in.
// These are the include guards used by MinGW's GL headers; defining
// them before the #include makes those headers skip their contents.
#define __gl_h_
#define __GL_H__
#define __glext_h_
#define __GLEXT_H__
// Windows font fallback — Consolas is the closest monospace to DejaVu.
// Override with VSRG_OVERLAY_FONT env var (same as Linux).
#define DEFAULT_FONT_PATH "C:\\Windows\\Fonts\\consola.ttf"

#include "render.c"
