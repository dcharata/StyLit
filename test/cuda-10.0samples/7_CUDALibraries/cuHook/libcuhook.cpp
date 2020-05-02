/*
* Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
*
* NOTICE TO USER:
*
* This source code is subject to NVIDIA ownership rights under U.S. and
* international Copyright laws.
*
* NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
* CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
* IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
* REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
* MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
* IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
* OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
* OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
* OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE
* OR PERFORMANCE OF THIS SOURCE CODE.
*
* U.S. Government End Users.  This source code is a "commercial item" as
* that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of
* "commercial computer software" and "commercial computer software
* documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995)
* and is provided to the U.S. Government only as a commercial end item.
* Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
* 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
* source code with only those rights set forth herein.
*/

// This sample demonstrates a simple library to interpose CUDA symbols

#define __USE_GNU
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <stdlib.h>

#include <cuda.h>
#include "libcuhook.h"

// For interposing dlsym(). See elf/dl-libc.c for the internal dlsym interface function
// For interposing dlopen(). Sell elf/dl-lib.c for the internal dlopen_mode interface function
extern "C" { void* __libc_dlsym (void *map, const char *name); }
extern "C" { void* __libc_dlopen_mode (const char* name, int mode); }

// We need to give the pre-processor a chance to replace a function, such as:
// cuMemAlloc => cuMemAlloc_v2
#define STRINGIFY(x) #x
#define CUDA_SYMBOL_STRING(x) STRINGIFY(x)

// We need to interpose dlsym since anyone using dlopen+dlsym to get the CUDA driver symbols will bypass
// the hooking mechanism (this includes the CUDA runtime). Its tricky though, since if we replace the
// real dlsym with ours, we can't dlsym() the real dlsym. To get around that, call the 'private'
// libc interface called __libc_dlsym to get the real dlsym.
typedef void* (*fnDlsym)(void*, const char*);

static void* real_dlsym(void *handle, const char* symbol)
{
    static fnDlsym internal_dlsym = (fnDlsym)__libc_dlsym(__libc_dlopen_mode("libdl.so.2", RTLD_LAZY), "dlsym");
    return (*internal_dlsym)(handle, symbol);
}

// Main structure that gets initialized at library load time
// Choose a unique name, or it can clash with other preloaded libraries.
struct cuHookInfo
{
    void        *handle;
    void        *preHooks[CU_HOOK_SYMBOLS];
    void        *postHooks[CU_HOOK_SYMBOLS];

    // Debugging/Stats Info
    int         bDebugEnabled;
    int         hookedFunctionCalls[CU_HOOK_SYMBOLS];

    cuHookInfo()
    {
        const char* envHookDebug;

        // Check environment for CU_HOOK_DEBUG to facilitate debugging
        envHookDebug = getenv("CU_HOOK_DEBUG");
        if (envHookDebug && envHookDebug[0] == '1') {
            bDebugEnabled = 1;
            fprintf(stderr, "* %6d >> CUDA HOOK Library loaded.\n", getpid());
        }
    }

    ~cuHookInfo()
    {
        if (bDebugEnabled) {
            pid_t pid = getpid();
            // You can gather statistics, timings, etc.
            fprintf(stderr, "* %6d >> CUDA HOOK Library Unloaded - Statistics:\n", pid);
            fprintf(stderr, "* %6d >> %20s ... %d\n", pid,
                    CUDA_SYMBOL_STRING(cuMemAlloc), hookedFunctionCalls[CU_HOOK_MEM_ALLOC]);
            fprintf(stderr, "* %6d >> %20s ... %d\n", pid,
                    CUDA_SYMBOL_STRING(cuMemFree), hookedFunctionCalls[CU_HOOK_MEM_FREE]);
            fprintf(stderr, "* %6d >> %20s ... %d\n", pid,
                    CUDA_SYMBOL_STRING(cuCtxGetCurrent), hookedFunctionCalls[CU_HOOK_CTX_GET_CURRENT]);
            fprintf(stderr, "* %6d >> %20s ... %d\n", pid,
                    CUDA_SYMBOL_STRING(cuCtxSetCurrent), hookedFunctionCalls[CU_HOOK_CTX_SET_CURRENT]);
            fprintf(stderr, "* %6d >> %20s ... %d\n", pid,
                    CUDA_SYMBOL_STRING(cuCtxDestroy), hookedFunctionCalls[CU_HOOK_CTX_DESTROY]);
        }
        if (handle) {
            dlclose(handle);
        }
    }

};

static struct cuHookInfo cuhl;

// Exposed API
void cuHookRegisterCallback(HookSymbols symbol, HookTypes type, void* callback)
{
    if (type == PRE_CALL_HOOK) {
        cuhl.preHooks[symbol] = callback;
    }
    else if (type == POST_CALL_HOOK) {
        cuhl.postHooks[symbol] = callback;
    }
}

/*
 ** Interposed Functions
 */
void* dlsym(void *handle, const char *symbol)
{
    // Early out if not a CUDA driver symbol
    if (strncmp(symbol, "cu", 2) != 0) {
        return (real_dlsym(handle, symbol));
    }

    if (strcmp(symbol, CUDA_SYMBOL_STRING(cuMemAlloc)) == 0) {
        return (void*)(&cuMemAlloc);
    }
    else if (strcmp(symbol, CUDA_SYMBOL_STRING(cuMemFree)) == 0) {
        return (void*)(&cuMemFree);
    }
    else if (strcmp(symbol, CUDA_SYMBOL_STRING(cuCtxGetCurrent)) == 0) {
        return (void*)(&cuCtxGetCurrent);
    }
    else if (strcmp(symbol, CUDA_SYMBOL_STRING(cuCtxSetCurrent)) == 0) {
        return (void*)(&cuCtxSetCurrent);
    }
    else if (strcmp(symbol, CUDA_SYMBOL_STRING(cuCtxDestroy)) == 0) {
        return (void*)(&cuCtxDestroy);
    }
    return (real_dlsym(handle, symbol));
}

/*
** If the user of this library does not wish to include CUDA specific code/headers in the code,
** then all the parameters can be changed and/or simply casted before calling the callback.
*/
#define CU_HOOK_GENERATE_INTERCEPT(hooksymbol, funcname, params, ...)   \
    CUresult CUDAAPI funcname params                                    \
    {                                                                   \
        static void* real_func = (void*)real_dlsym(RTLD_NEXT, CUDA_SYMBOL_STRING(funcname)); \
        CUresult result = CUDA_SUCCESS;                                 \
                                                                        \
        if (cuhl.bDebugEnabled) {                                       \
            cuhl.hookedFunctionCalls[hooksymbol]++;                     \
        }                                                               \
        if (cuhl.preHooks[hooksymbol]) {                                \
            ((CUresult CUDAAPI (*)params)cuhl.preHooks[hooksymbol])(__VA_ARGS__);               \
        }                                                                                       \
        result = ((CUresult CUDAAPI (*)params)real_func)(__VA_ARGS__);                          \
        if (cuhl.postHooks[hooksymbol] && result == CUDA_SUCCESS) {                             \
            ((CUresult CUDAAPI (*)params)cuhl.postHooks[hooksymbol])(__VA_ARGS__);              \
        }                                                                                       \
        return (result);                                                                        \
    }

CU_HOOK_GENERATE_INTERCEPT(CU_HOOK_MEM_ALLOC, cuMemAlloc, (CUdeviceptr *dptr, size_t bytesize), dptr, bytesize)
CU_HOOK_GENERATE_INTERCEPT(CU_HOOK_MEM_FREE, cuMemFree, (CUdeviceptr dptr), dptr)
CU_HOOK_GENERATE_INTERCEPT(CU_HOOK_CTX_GET_CURRENT, cuCtxGetCurrent, (CUcontext *pctx), pctx)
CU_HOOK_GENERATE_INTERCEPT(CU_HOOK_CTX_SET_CURRENT, cuCtxSetCurrent, (CUcontext ctx), ctx)
CU_HOOK_GENERATE_INTERCEPT(CU_HOOK_CTX_DESTROY, cuCtxDestroy, (CUcontext ctx), ctx)
