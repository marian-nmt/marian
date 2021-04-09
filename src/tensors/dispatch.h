#pragma once

#ifdef CUDA_FOUND

#define DISPATCH1(Function, Arg1)                                 \
  namespace gpu {                                                 \
  void Function(Arg1);                                            \
  }                                                               \
  namespace cpu {                                                 \
  void Function(Arg1);                                            \
  }                                                               \
  void Function(Arg1 arg1) {                                      \
    if(arg1->getBackend()->getDeviceId().type == DeviceType::gpu) \
      gpu::Function(arg1);                                        \
    else                                                          \
      cpu::Function(arg1);                                        \
  }

#define DISPATCH2(Function, Arg1, Arg2)                           \
  namespace gpu {                                                 \
  void Function(Arg1, Arg2);                                      \
  }                                                               \
  namespace cpu {                                                 \
  void Function(Arg1, Arg2);                                      \
  }                                                               \
  static inline void Function(Arg1 arg1, Arg2 arg2) {             \
    if(arg1->getBackend()->getDeviceId().type == DeviceType::gpu) \
      gpu::Function(arg1, arg2);                                  \
    else                                                          \
      cpu::Function(arg1, arg2);                                  \
  }

#define DISPATCH3(Function, Arg1, Arg2, Arg3)                     \
  namespace gpu {                                                 \
  void Function(Arg1, Arg2, Arg3);                                \
  }                                                               \
  namespace cpu {                                                 \
  void Function(Arg1, Arg2, Arg3);                                \
  }                                                               \
  static inline void Function(Arg1 arg1, Arg2 arg2, Arg3 arg3) {  \
    if(arg1->getBackend()->getDeviceId().type == DeviceType::gpu) \
      gpu::Function(arg1, arg2, arg3);                            \
    else                                                          \
      cpu::Function(arg1, arg2, arg3);                            \
  }

#define DISPATCH4(Function, Arg1, Arg2, Arg3, Arg4)                         \
  namespace gpu {                                                           \
  void Function(Arg1, Arg2, Arg3, Arg4);                                    \
  }                                                                         \
  namespace cpu {                                                           \
  void Function(Arg1, Arg2, Arg3, Arg4);                                    \
  }                                                                         \
  static inline void Function(Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4) { \
    if(arg1->getBackend()->getDeviceId().type == DeviceType::gpu)           \
      gpu::Function(arg1, arg2, arg3, arg4);                                \
    else                                                                    \
      cpu::Function(arg1, arg2, arg3, arg4);                                \
  }

#define DISPATCH5(Function, Arg1, Arg2, Arg3, Arg4, Arg5)         \
  namespace gpu {                                                 \
  void Function(Arg1, Arg2, Arg3, Arg4, Arg5);                    \
  }                                                               \
  namespace cpu {                                                 \
  void Function(Arg1, Arg2, Arg3, Arg4, Arg5);                    \
  }                                                               \
  static inline void Function(                                    \
      Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4, Arg5 arg5) {    \
    if(arg1->getBackend()->getDeviceId().type == DeviceType::gpu) \
      gpu::Function(arg1, arg2, arg3, arg4, arg5);                \
    else                                                          \
      cpu::Function(arg1, arg2, arg3, arg4, arg5);                \
  }

#define DISPATCH6(Function, Arg1, Arg2, Arg3, Arg4, Arg5, Arg6)           \
  namespace gpu {                                                         \
  void Function(Arg1, Arg2, Arg3, Arg4, Arg5, Arg6);                      \
  }                                                                       \
  namespace cpu {                                                         \
  void Function(Arg1, Arg2, Arg3, Arg4, Arg5, Arg6);                      \
  }                                                                       \
  static inline void Function(                                            \
      Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4, Arg5 arg5, Arg6 arg6) { \
    if(arg1->getBackend()->getDeviceId().type == DeviceType::gpu)         \
      gpu::Function(arg1, arg2, arg3, arg4, arg5, arg6);                  \
    else                                                                  \
      cpu::Function(arg1, arg2, arg3, arg4, arg5, arg6);                  \
  }

#define DISPATCH7(Function, Arg1, Arg2, Arg3, Arg4, Arg5, Arg6, Arg7) \
  namespace gpu {                                                     \
  void Function(Arg1, Arg2, Arg3, Arg4, Arg5, Arg6, Arg7);            \
  }                                                                   \
  namespace cpu {                                                     \
  void Function(Arg1, Arg2, Arg3, Arg4, Arg5, Arg6, Arg7);            \
  }                                                                   \
  static inline void Function(Arg1 arg1,                              \
                              Arg2 arg2,                              \
                              Arg3 arg3,                              \
                              Arg4 arg4,                              \
                              Arg5 arg5,                              \
                              Arg6 arg6,                              \
                              Arg7 arg7) {                            \
    if(arg1->getBackend()->getDeviceId().type == DeviceType::gpu)     \
      gpu::Function(arg1, arg2, arg3, arg4, arg5, arg6, arg7);        \
    else                                                              \
      cpu::Function(arg1, arg2, arg3, arg4, arg5, arg6, arg7);        \
  }

#define DISPATCH8(Function, Arg1, Arg2, Arg3, Arg4, Arg5, Arg6, Arg7, Arg8) \
  namespace gpu {                                                           \
  void Function(Arg1, Arg2, Arg3, Arg4, Arg5, Arg6, Arg7, Arg8);            \
  }                                                                         \
  namespace cpu {                                                           \
  void Function(Arg1, Arg2, Arg3, Arg4, Arg5, Arg6, Arg7, Arg8);            \
  }                                                                         \
  static inline void Function(Arg1 arg1,                                    \
                              Arg2 arg2,                                    \
                              Arg3 arg3,                                    \
                              Arg4 arg4,                                    \
                              Arg5 arg5,                                    \
                              Arg6 arg6,                                    \
                              Arg7 arg7,                                    \
                              Arg8 arg8) {                                  \
    if(arg1->getBackend()->getDeviceId().type == DeviceType::gpu)           \
      gpu::Function(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8);        \
    else                                                                    \
      cpu::Function(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8);        \
  }

#define DISPATCH9(                                                         \
    Function, Arg1, Arg2, Arg3, Arg4, Arg5, Arg6, Arg7, Arg8, Arg9)        \
  namespace gpu {                                                          \
  void Function(Arg1, Arg2, Arg3, Arg4, Arg5, Arg6, Arg7, Arg8, Arg9);     \
  }                                                                        \
  namespace cpu {                                                          \
  void Function(Arg1, Arg2, Arg3, Arg4, Arg5, Arg6, Arg7, Arg8, Arg9);     \
  }                                                                        \
  static inline void Function(Arg1 arg1,                                   \
                              Arg2 arg2,                                   \
                              Arg3 arg3,                                   \
                              Arg4 arg4,                                   \
                              Arg5 arg5,                                   \
                              Arg6 arg6,                                   \
                              Arg7 arg7,                                   \
                              Arg8 arg8,                                   \
                              Arg9 arg9) {                                 \
    if(arg1->getBackend()->getDeviceId().type == DeviceType::gpu)          \
      gpu::Function(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9); \
    else                                                                   \
      cpu::Function(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9); \
  }

#define DISPATCH10(                                                             \
  Function, Arg1, Arg2, Arg3, Arg4, Arg5, Arg6, Arg7, Arg8, Arg9, Arg10)        \
namespace gpu {                                                                 \
void Function(Arg1, Arg2, Arg3, Arg4, Arg5, Arg6, Arg7, Arg8, Arg9, Arg10);     \
}                                                                               \
namespace cpu {                                                                 \
void Function(Arg1, Arg2, Arg3, Arg4, Arg5, Arg6, Arg7, Arg8, Arg9, Arg10);     \
}                                                                               \
static inline void Function(Arg1 arg1,                                          \
                            Arg2 arg2,                                          \
                            Arg3 arg3,                                          \
                            Arg4 arg4,                                          \
                            Arg5 arg5,                                          \
                            Arg6 arg6,                                          \
                            Arg7 arg7,                                          \
                            Arg8 arg8,                                          \
                            Arg9 arg9,                                          \
                            Arg10 arg10) {                                      \
  if(arg1->getBackend()->getDeviceId().type == DeviceType::gpu)                 \
    gpu::Function(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10); \
  else                                                                          \
    cpu::Function(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10); \
}

#else

#define DISPATCH1(Function, Arg1) \
  namespace cpu {                 \
  void Function(Arg1);            \
  }                               \
  void Function(Arg1 arg1) { cpu::Function(arg1); }

#define DISPATCH2(Function, Arg1, Arg2)               \
  namespace cpu {                                     \
  void Function(Arg1, Arg2);                          \
  }                                                   \
  static inline void Function(Arg1 arg1, Arg2 arg2) { \
    cpu::Function(arg1, arg2);                        \
  }

#define DISPATCH3(Function, Arg1, Arg2, Arg3)                    \
  namespace cpu {                                                \
  void Function(Arg1, Arg2, Arg3);                               \
  }                                                              \
  static inline void Function(Arg1 arg1, Arg2 arg2, Arg3 arg3) { \
    cpu::Function(arg1, arg2, arg3);                             \
  }

#define DISPATCH4(Function, Arg1, Arg2, Arg3, Arg4)                         \
  namespace cpu {                                                           \
  void Function(Arg1, Arg2, Arg3, Arg4);                                    \
  }                                                                         \
  static inline void Function(Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4) { \
    cpu::Function(arg1, arg2, arg3, arg4);                                  \
  }

#define DISPATCH5(Function, Arg1, Arg2, Arg3, Arg4, Arg5)      \
  namespace cpu {                                              \
  void Function(Arg1, Arg2, Arg3, Arg4, Arg5);                 \
  }                                                            \
  static inline void Function(                                 \
      Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4, Arg5 arg5) { \
    cpu::Function(arg1, arg2, arg3, arg4, arg5);               \
  }

#define DISPATCH6(Function, Arg1, Arg2, Arg3, Arg4, Arg5, Arg6)           \
  namespace cpu {                                                         \
  void Function(Arg1, Arg2, Arg3, Arg4, Arg5, Arg6);                      \
  }                                                                       \
  static inline void Function(                                            \
      Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4, Arg5 arg5, Arg6 arg6) { \
    cpu::Function(arg1, arg2, arg3, arg4, arg5, arg6);                    \
  }

#define DISPATCH7(Function, Arg1, Arg2, Arg3, Arg4, Arg5, Arg6, Arg7) \
  namespace cpu {                                                     \
  void Function(Arg1, Arg2, Arg3, Arg4, Arg5, Arg6, Arg7);            \
  }                                                                   \
  static inline void Function(Arg1 arg1,                              \
                              Arg2 arg2,                              \
                              Arg3 arg3,                              \
                              Arg4 arg4,                              \
                              Arg5 arg5,                              \
                              Arg6 arg6,                              \
                              Arg7 arg7) {                            \
    cpu::Function(arg1, arg2, arg3, arg4, arg5, arg6, arg7);          \
  }

#define DISPATCH8(Function, Arg1, Arg2, Arg3, Arg4, Arg5, Arg6, Arg7, Arg8) \
  namespace cpu {                                                           \
  void Function(Arg1, Arg2, Arg3, Arg4, Arg5, Arg6, Arg7, Arg8);            \
  }                                                                         \
  static inline void Function(Arg1 arg1,                                    \
                              Arg2 arg2,                                    \
                              Arg3 arg3,                                    \
                              Arg4 arg4,                                    \
                              Arg5 arg5,                                    \
                              Arg6 arg6,                                    \
                              Arg7 arg7,                                    \
                              Arg8 arg8) {                                  \
    cpu::Function(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8);          \
  }

#define DISPATCH9(                                                       \
    Function, Arg1, Arg2, Arg3, Arg4, Arg5, Arg6, Arg7, Arg8, Arg9)      \
  namespace cpu {                                                        \
  void Function(Arg1, Arg2, Arg3, Arg4, Arg5, Arg6, Arg7, Arg8, Arg9);   \
  }                                                                      \
  static inline void Function(Arg1 arg1,                                 \
                              Arg2 arg2,                                 \
                              Arg3 arg3,                                 \
                              Arg4 arg4,                                 \
                              Arg5 arg5,                                 \
                              Arg6 arg6,                                 \
                              Arg7 arg7,                                 \
                              Arg8 arg8,                                 \
                              Arg9 arg9) {                               \
    cpu::Function(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9); \
  }

#define DISPATCH10(                                                             \
    Function, Arg1, Arg2, Arg3, Arg4, Arg5, Arg6, Arg7, Arg8, Arg9, Arg10)      \
  namespace cpu {                                                               \
  void Function(Arg1, Arg2, Arg3, Arg4, Arg5, Arg6, Arg7, Arg8, Arg9, Arg10);   \
  }                                                                             \
  static inline void Function(Arg1 arg1,                                        \
                              Arg2 arg2,                                        \
                              Arg3 arg3,                                        \
                              Arg4 arg4,                                        \
                              Arg5 arg5,                                        \
                              Arg6 arg6,                                        \
                              Arg7 arg7,                                        \
                              Arg8 arg8,                                        \
                              Arg9 arg9,                                        \
                              Arg10 arg10) {                                    \
    cpu::Function(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10); \
  }

#endif
