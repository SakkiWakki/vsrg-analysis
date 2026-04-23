// Force-included on MSVC via /FI so the Linux overlay sources that use
// GCC's __atomic_* builtins (widgets.c, potentially others) compile
// unchanged. x86/x64 aligned stores of the integer sizes used here
// are already atomic; MemoryBarrier() gives a full fence where
// __ATOMIC_RELEASE is requested.

#pragma once
#ifdef _MSC_VER

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <intrin.h>

#define __ATOMIC_RELAXED 0
#define __ATOMIC_CONSUME 1
#define __ATOMIC_ACQUIRE 2
#define __ATOMIC_RELEASE 3
#define __ATOMIC_ACQ_REL 4
#define __ATOMIC_SEQ_CST 5

#define __atomic_store_n(ptr, val, order) \
    ((void)(*(ptr) = (val)), MemoryBarrier())

#define __atomic_load_n(ptr, order) (*(ptr))

#define __atomic_add_fetch(ptr, val, order) \
    (_InterlockedExchangeAdd((volatile long *)(ptr), (long)(val)) + (long)(val))

#define __atomic_thread_fence(order) MemoryBarrier()

#endif
