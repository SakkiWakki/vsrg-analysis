// Windows OpenGL extension loader for NanoVG GL2.
//
// Hand-rolled wglGetProcAddress loader.
// Include this before nanovg_gl.h. After vsrg_gl_load_extensions()
// every GL extension below is reachable via its standard name; the
// #defines at the bottom redirect calls through our loaded pointers
// transparently so nanovg_gl.h and render.c compile unchanged.
//
// Must be called with a GL context current (i.e. on first swap frame,
// not in DllMain).

#pragma once
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <GL/gl.h>

// Types missing from the Windows SDK's gl.h baseline.
typedef char  GLchar;
typedef ptrdiff_t GLsizeiptr;

// --- function pointer typedefs ---

typedef void   (APIENTRY *PFNGLACTIVETEXTUREPROC)             (GLenum);
typedef void   (APIENTRY *PFNGLATTACHSHADERPROC)              (GLuint, GLuint);
typedef void   (APIENTRY *PFNGLBLENDFUNCSEPARATEPROC)         (GLenum, GLenum, GLenum, GLenum);
typedef void   (APIENTRY *PFNGLBINDATTRIBLOCATIONPROC)        (GLuint, GLuint, const GLchar *);
typedef void   (APIENTRY *PFNGLBINDBUFFERPROC)                (GLenum, GLuint);
typedef void   (APIENTRY *PFNGLBUFFERDATAPROC)                (GLenum, GLsizeiptr, const void *, GLenum);
typedef void   (APIENTRY *PFNGLCOMPILESHADERPROC)             (GLuint);
typedef GLuint (APIENTRY *PFNGLCREATEPROGRAMPROC)             (void);
typedef GLuint (APIENTRY *PFNGLCREATESHADERPROC)              (GLenum);
typedef void   (APIENTRY *PFNGLDELETEBUFFERSPROC)             (GLsizei, const GLuint *);
typedef void   (APIENTRY *PFNGLDELETEPROGRAMPROC)             (GLuint);
typedef void   (APIENTRY *PFNGLDELETESHADERPROC)              (GLuint);
typedef void   (APIENTRY *PFNGLDETACHSHADERPROC)              (GLuint, GLuint);
typedef void   (APIENTRY *PFNGLDISABLEVERTEXATTRIBARRAYPROC)  (GLuint);
typedef void   (APIENTRY *PFNGLENABLEVERTEXATTRIBARRAYPROC)   (GLuint);
typedef void   (APIENTRY *PFNGLGENBUFFERSPROC)                (GLsizei, GLuint *);
typedef void   (APIENTRY *PFNGLGENERATEMIPMAPPROC)            (GLenum);
typedef void   (APIENTRY *PFNGLGETPROGRAMINFOLOGPROC)         (GLuint, GLsizei, GLsizei *, GLchar *);
typedef void   (APIENTRY *PFNGLGETPROGRAMIVPROC)              (GLuint, GLenum, GLint *);
typedef void   (APIENTRY *PFNGLGETSHADERINFOLOGPROC)          (GLuint, GLsizei, GLsizei *, GLchar *);
typedef void   (APIENTRY *PFNGLGETSHADERIVPROC)               (GLuint, GLenum, GLint *);
typedef GLint  (APIENTRY *PFNGLGETUNIFORMLOCATIONPROC)        (GLuint, const GLchar *);
typedef void   (APIENTRY *PFNGLLINKPROGRAMPROC)               (GLuint);
typedef void   (APIENTRY *PFNGLSHADERSOURCEPROC)              (GLuint, GLsizei, const GLchar *const *, const GLint *);
typedef void   (APIENTRY *PFNGLSTENCILFUNCSEPARATEPROC)       (GLenum, GLenum, GLint, GLuint);
typedef void   (APIENTRY *PFNGLSTENCILMASKSEPARATEPROC)       (GLenum, GLuint);
typedef void   (APIENTRY *PFNGLSTENCILOPSEPARATEPROC)         (GLenum, GLenum, GLenum, GLenum);
typedef void   (APIENTRY *PFNGLUNIFORM1IPROC)                 (GLint, GLint);
typedef void   (APIENTRY *PFNGLUNIFORM2FVPROC)                (GLint, GLsizei, const GLfloat *);
typedef void   (APIENTRY *PFNGLUNIFORM4FVPROC)                (GLint, GLsizei, const GLfloat *);
typedef void   (APIENTRY *PFNGLUSEPROGRAMPROC)                (GLuint);
typedef void   (APIENTRY *PFNGLVERTEXATTRIBPOINTERPROC)       (GLuint, GLint, GLenum, GLboolean, GLsizei, const void *);
typedef void   (APIENTRY *PFNGLTEXSUBIMAGE2DPROC)             (GLenum, GLint, GLint, GLint, GLsizei, GLsizei, GLenum, GLenum, const void *);

// --- extern declarations ---

extern PFNGLACTIVETEXTUREPROC            vsrg_glActiveTexture;
extern PFNGLATTACHSHADERPROC             vsrg_glAttachShader;
extern PFNGLBLENDFUNCSEPARATEPROC        vsrg_glBlendFuncSeparate;
extern PFNGLBINDATTRIBLOCATIONPROC       vsrg_glBindAttribLocation;
extern PFNGLBINDBUFFERPROC               vsrg_glBindBuffer;
extern PFNGLBUFFERDATAPROC               vsrg_glBufferData;
extern PFNGLCOMPILESHADERPROC            vsrg_glCompileShader;
extern PFNGLCREATEPROGRAMPROC            vsrg_glCreateProgram;
extern PFNGLCREATESHADERPROC             vsrg_glCreateShader;
extern PFNGLDELETEBUFFERSPROC            vsrg_glDeleteBuffers;
extern PFNGLDELETEPROGRAMPROC            vsrg_glDeleteProgram;
extern PFNGLDELETESHADERPROC             vsrg_glDeleteShader;
extern PFNGLDETACHSHADERPROC             vsrg_glDetachShader;
extern PFNGLDISABLEVERTEXATTRIBARRAYPROC vsrg_glDisableVertexAttribArray;
extern PFNGLENABLEVERTEXATTRIBARRAYPROC  vsrg_glEnableVertexAttribArray;
extern PFNGLGENBUFFERSPROC               vsrg_glGenBuffers;
extern PFNGLGENERATEMIPMAPPROC           vsrg_glGenerateMipmap;
extern PFNGLGETPROGRAMINFOLOGPROC        vsrg_glGetProgramInfoLog;
extern PFNGLGETPROGRAMIVPROC             vsrg_glGetProgramiv;
extern PFNGLGETSHADERINFOLOGPROC         vsrg_glGetShaderInfoLog;
extern PFNGLGETSHADERIVPROC              vsrg_glGetShaderiv;
extern PFNGLGETUNIFORMLOCATIONPROC       vsrg_glGetUniformLocation;
extern PFNGLLINKPROGRAMPROC              vsrg_glLinkProgram;
extern PFNGLSHADERSOURCEPROC             vsrg_glShaderSource;
extern PFNGLSTENCILFUNCSEPARATEPROC      vsrg_glStencilFuncSeparate;
extern PFNGLSTENCILMASKSEPARATEPROC      vsrg_glStencilMaskSeparate;
extern PFNGLSTENCILOPSEPARATEPROC        vsrg_glStencilOpSeparate;
extern PFNGLUNIFORM1IPROC                vsrg_glUniform1i;
extern PFNGLUNIFORM2FVPROC               vsrg_glUniform2fv;
extern PFNGLUNIFORM4FVPROC               vsrg_glUniform4fv;
extern PFNGLUSEPROGRAMPROC               vsrg_glUseProgram;
extern PFNGLVERTEXATTRIBPOINTERPROC      vsrg_glVertexAttribPointer;
extern PFNGLTEXSUBIMAGE2DPROC            vsrg_glTexSubImage2D;

// --- name redirects (transparent to nanovg_gl.h and render.c) ---

#define glActiveTexture             vsrg_glActiveTexture
#define glAttachShader              vsrg_glAttachShader
#define glBlendFuncSeparate         vsrg_glBlendFuncSeparate
#define glBindAttribLocation        vsrg_glBindAttribLocation
#define glBindBuffer                vsrg_glBindBuffer
#define glBufferData                vsrg_glBufferData
#define glCompileShader             vsrg_glCompileShader
#define glCreateProgram             vsrg_glCreateProgram
#define glCreateShader              vsrg_glCreateShader
#define glDeleteBuffers             vsrg_glDeleteBuffers
#define glDeleteProgram             vsrg_glDeleteProgram
#define glDeleteShader              vsrg_glDeleteShader
#define glDetachShader              vsrg_glDetachShader
#define glDisableVertexAttribArray  vsrg_glDisableVertexAttribArray
#define glEnableVertexAttribArray   vsrg_glEnableVertexAttribArray
#define glGenBuffers                vsrg_glGenBuffers
#define glGenerateMipmap            vsrg_glGenerateMipmap
#define glGetProgramInfoLog         vsrg_glGetProgramInfoLog
#define glGetProgramiv              vsrg_glGetProgramiv
#define glGetShaderInfoLog          vsrg_glGetShaderInfoLog
#define glGetShaderiv               vsrg_glGetShaderiv
#define glGetUniformLocation        vsrg_glGetUniformLocation
#define glLinkProgram               vsrg_glLinkProgram
#define glShaderSource              vsrg_glShaderSource
#define glStencilFuncSeparate       vsrg_glStencilFuncSeparate
#define glStencilMaskSeparate       vsrg_glStencilMaskSeparate
#define glStencilOpSeparate         vsrg_glStencilOpSeparate
#define glUniform1i                 vsrg_glUniform1i
#define glUniform2fv                vsrg_glUniform2fv
#define glUniform4fv                vsrg_glUniform4fv
#define glUseProgram                vsrg_glUseProgram
#define glVertexAttribPointer       vsrg_glVertexAttribPointer
#define glTexSubImage2D             vsrg_glTexSubImage2D

// Needed by nanovg_gl.h constant references ; not in Windows SDK gl.h.
#define GL_TEXTURE0                     0x84C0u
#define GL_ARRAY_BUFFER                 0x8892u
#define GL_STREAM_DRAW                  0x88E0u
#define GL_FRAGMENT_SHADER              0x8B30u
#define GL_VERTEX_SHADER                0x8B31u
#define GL_COMPILE_STATUS               0x8B81u
#define GL_LINK_STATUS                  0x8B82u
#define GL_CLAMP_TO_EDGE                0x812Fu
#define GL_GENERATE_MIPMAP              0x8191u
#define GL_MIRRORED_REPEAT              0x8370u
#define GL_INCR_WRAP                    0x8507u
#define GL_DECR_WRAP                    0x8508u

#ifdef __cplusplus
extern "C" {
#endif

// Load all extension pointers above. Must be called with a GL context
// current. Returns 1 if all critical functions loaded, 0 on failure.
int vsrg_gl_load_extensions(void);

#ifdef __cplusplus
}
#endif
