#include "win_gl_loader.h"

#include <stdio.h>

// --- function pointer definitions ---

PFNGLACTIVETEXTUREPROC            vsrg_glActiveTexture            = NULL;
PFNGLATTACHSHADERPROC             vsrg_glAttachShader             = NULL;
PFNGLBLENDFUNCSEPARATEPROC        vsrg_glBlendFuncSeparate        = NULL;
PFNGLBINDATTRIBLOCATIONPROC       vsrg_glBindAttribLocation       = NULL;
PFNGLBINDBUFFERPROC               vsrg_glBindBuffer               = NULL;
PFNGLBUFFERDATAPROC               vsrg_glBufferData               = NULL;
PFNGLCOMPILESHADERPROC            vsrg_glCompileShader            = NULL;
PFNGLCREATEPROGRAMPROC            vsrg_glCreateProgram            = NULL;
PFNGLCREATESHADERPROC             vsrg_glCreateShader             = NULL;
PFNGLDELETEBUFFERSPROC            vsrg_glDeleteBuffers            = NULL;
PFNGLDELETEPROGRAMPROC            vsrg_glDeleteProgram            = NULL;
PFNGLDELETESHADERPROC             vsrg_glDeleteShader             = NULL;
PFNGLDETACHSHADERPROC             vsrg_glDetachShader             = NULL;
PFNGLDISABLEVERTEXATTRIBARRAYPROC vsrg_glDisableVertexAttribArray = NULL;
PFNGLENABLEVERTEXATTRIBARRAYPROC  vsrg_glEnableVertexAttribArray  = NULL;
PFNGLGENBUFFERSPROC               vsrg_glGenBuffers               = NULL;
PFNGLGENERATEMIPMAPPROC           vsrg_glGenerateMipmap           = NULL;
PFNGLGETPROGRAMINFOLOGPROC        vsrg_glGetProgramInfoLog        = NULL;
PFNGLGETPROGRAMIVPROC             vsrg_glGetProgramiv             = NULL;
PFNGLGETSHADERINFOLOGPROC         vsrg_glGetShaderInfoLog         = NULL;
PFNGLGETSHADERIVPROC              vsrg_glGetShaderiv              = NULL;
PFNGLGETUNIFORMLOCATIONPROC       vsrg_glGetUniformLocation       = NULL;
PFNGLLINKPROGRAMPROC              vsrg_glLinkProgram              = NULL;
PFNGLSHADERSOURCEPROC             vsrg_glShaderSource             = NULL;
PFNGLSTENCILFUNCSEPARATEPROC      vsrg_glStencilFuncSeparate      = NULL;
PFNGLSTENCILMASKSEPARATEPROC      vsrg_glStencilMaskSeparate      = NULL;
PFNGLSTENCILOPSEPARATEPROC        vsrg_glStencilOpSeparate        = NULL;
PFNGLUNIFORM1IPROC                vsrg_glUniform1i                = NULL;
PFNGLUNIFORM2FVPROC               vsrg_glUniform2fv               = NULL;
PFNGLUNIFORM4FVPROC               vsrg_glUniform4fv               = NULL;
PFNGLUSEPROGRAMPROC               vsrg_glUseProgram               = NULL;
PFNGLVERTEXATTRIBPOINTERPROC      vsrg_glVertexAttribPointer      = NULL;
PFNGLTEXSUBIMAGE2DPROC            vsrg_glTexSubImage2D            = NULL;

#define LOAD(TYPE, name)                                                       \
    do {                                                                       \
        vsrg_##name = (TYPE)wglGetProcAddress(#name);                          \
        if (!vsrg_##name) {                                                    \
            fprintf(stderr, "[vsrg-gl-loader] failed to load " #name "\n");    \
            ok = 0;                                                            \
        }                                                                      \
    } while (0)

int vsrg_gl_load_extensions(void) {
    int ok = 1;

    LOAD(PFNGLACTIVETEXTUREPROC,            glActiveTexture);
    LOAD(PFNGLATTACHSHADERPROC,             glAttachShader);
    LOAD(PFNGLBLENDFUNCSEPARATEPROC,        glBlendFuncSeparate);
    LOAD(PFNGLBINDATTRIBLOCATIONPROC,       glBindAttribLocation);
    LOAD(PFNGLBINDBUFFERPROC,               glBindBuffer);
    LOAD(PFNGLBUFFERDATAPROC,               glBufferData);
    LOAD(PFNGLCOMPILESHADERPROC,            glCompileShader);
    LOAD(PFNGLCREATEPROGRAMPROC,            glCreateProgram);
    LOAD(PFNGLCREATESHADERPROC,             glCreateShader);
    LOAD(PFNGLDELETEBUFFERSPROC,            glDeleteBuffers);
    LOAD(PFNGLDELETEPROGRAMPROC,            glDeleteProgram);
    LOAD(PFNGLDELETESHADERPROC,             glDeleteShader);
    LOAD(PFNGLDETACHSHADERPROC,             glDetachShader);
    LOAD(PFNGLDISABLEVERTEXATTRIBARRAYPROC, glDisableVertexAttribArray);
    LOAD(PFNGLENABLEVERTEXATTRIBARRAYPROC,  glEnableVertexAttribArray);
    LOAD(PFNGLGENBUFFERSPROC,               glGenBuffers);
    LOAD(PFNGLGENERATEMIPMAPPROC,           glGenerateMipmap);
    LOAD(PFNGLGETPROGRAMINFOLOGPROC,        glGetProgramInfoLog);
    LOAD(PFNGLGETPROGRAMIVPROC,             glGetProgramiv);
    LOAD(PFNGLGETSHADERINFOLOGPROC,         glGetShaderInfoLog);
    LOAD(PFNGLGETSHADERIVPROC,              glGetShaderiv);
    LOAD(PFNGLGETUNIFORMLOCATIONPROC,       glGetUniformLocation);
    LOAD(PFNGLLINKPROGRAMPROC,              glLinkProgram);
    LOAD(PFNGLSHADERSOURCEPROC,             glShaderSource);
    LOAD(PFNGLSTENCILFUNCSEPARATEPROC,      glStencilFuncSeparate);
    LOAD(PFNGLSTENCILMASKSEPARATEPROC,      glStencilMaskSeparate);
    LOAD(PFNGLSTENCILOPSEPARATEPROC,        glStencilOpSeparate);
    LOAD(PFNGLUNIFORM1IPROC,                glUniform1i);
    LOAD(PFNGLUNIFORM2FVPROC,               glUniform2fv);
    LOAD(PFNGLUNIFORM4FVPROC,               glUniform4fv);
    LOAD(PFNGLUSEPROGRAMPROC,               glUseProgram);
    LOAD(PFNGLVERTEXATTRIBPOINTERPROC,      glVertexAttribPointer);
    LOAD(PFNGLTEXSUBIMAGE2DPROC,            glTexSubImage2D);

    return ok;
}
