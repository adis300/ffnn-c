/* Generated by the protocol buffer compiler.  DO NOT EDIT! */
/* Generated from: network.proto */

#ifndef PROTOBUF_C_network_2eproto__INCLUDED
#define PROTOBUF_C_network_2eproto__INCLUDED

#include <protobuf-c/protobuf-c.h>

PROTOBUF_C__BEGIN_DECLS

#if PROTOBUF_C_VERSION_NUMBER < 1003000
# error This file was generated by a newer version of protoc-c which is incompatible with your libprotobuf-c headers. Please update your headers.
#elif 1003000 < PROTOBUF_C_MIN_COMPILER_VERSION
# error This file was generated by an older version of protoc-c which is incompatible with your libprotobuf-c headers. Please regenerate this file with a newer version of protoc-c.
#endif


typedef struct _Ffnn__Weight Ffnn__Weight;
typedef struct _Ffnn__Bias Ffnn__Bias;
typedef struct _Ffnn__Network Ffnn__Network;


/* --- enums --- */

typedef enum _Ffnn__Network__ActivationType {
  FFNN__NETWORK__ACTIVATION_TYPE__SIGMOID = 0,
  FFNN__NETWORK__ACTIVATION_TYPE__LINEAR = 1,
  FFNN__NETWORK__ACTIVATION_TYPE__RELU = 2,
  FFNN__NETWORK__ACTIVATION_TYPE__THRESHOLD = 3,
  FFNN__NETWORK__ACTIVATION_TYPE__SOFTMAX = 4
    PROTOBUF_C__FORCE_ENUM_TO_BE_INT_SIZE(FFNN__NETWORK__ACTIVATION_TYPE)
} Ffnn__Network__ActivationType;

/* --- messages --- */

/*
 * [START messages]
 */
struct  _Ffnn__Weight
{
  ProtobufCMessage base;
  int32_t col;
  int32_t row;
  size_t n_grid;
  double *grid;
};
#define FFNN__WEIGHT__INIT \
 { PROTOBUF_C_MESSAGE_INIT (&ffnn__weight__descriptor) \
    , 0, 0, 0,NULL }


struct  _Ffnn__Bias
{
  ProtobufCMessage base;
  size_t n_vector;
  double *vector;
};
#define FFNN__BIAS__INIT \
 { PROTOBUF_C_MESSAGE_INIT (&ffnn__bias__descriptor) \
    , 0,NULL }


struct  _Ffnn__Network
{
  ProtobufCMessage base;
  size_t n_layersizes;
  int32_t *layersizes;
  size_t n_activations;
  Ffnn__Network__ActivationType *activations;
  size_t n_weights;
  Ffnn__Weight **weights;
  size_t n_biases;
  Ffnn__Bias **biases;
};
#define FFNN__NETWORK__INIT \
 { PROTOBUF_C_MESSAGE_INIT (&ffnn__network__descriptor) \
    , 0,NULL, 0,NULL, 0,NULL, 0,NULL }


/* Ffnn__Weight methods */
void   ffnn__weight__init
                     (Ffnn__Weight         *message);
size_t ffnn__weight__get_packed_size
                     (const Ffnn__Weight   *message);
size_t ffnn__weight__pack
                     (const Ffnn__Weight   *message,
                      uint8_t             *out);
size_t ffnn__weight__pack_to_buffer
                     (const Ffnn__Weight   *message,
                      ProtobufCBuffer     *buffer);
Ffnn__Weight *
       ffnn__weight__unpack
                     (ProtobufCAllocator  *allocator,
                      size_t               len,
                      const uint8_t       *data);
void   ffnn__weight__free_unpacked
                     (Ffnn__Weight *message,
                      ProtobufCAllocator *allocator);
/* Ffnn__Bias methods */
void   ffnn__bias__init
                     (Ffnn__Bias         *message);
size_t ffnn__bias__get_packed_size
                     (const Ffnn__Bias   *message);
size_t ffnn__bias__pack
                     (const Ffnn__Bias   *message,
                      uint8_t             *out);
size_t ffnn__bias__pack_to_buffer
                     (const Ffnn__Bias   *message,
                      ProtobufCBuffer     *buffer);
Ffnn__Bias *
       ffnn__bias__unpack
                     (ProtobufCAllocator  *allocator,
                      size_t               len,
                      const uint8_t       *data);
void   ffnn__bias__free_unpacked
                     (Ffnn__Bias *message,
                      ProtobufCAllocator *allocator);
/* Ffnn__Network methods */
void   ffnn__network__init
                     (Ffnn__Network         *message);
size_t ffnn__network__get_packed_size
                     (const Ffnn__Network   *message);
size_t ffnn__network__pack
                     (const Ffnn__Network   *message,
                      uint8_t             *out);
size_t ffnn__network__pack_to_buffer
                     (const Ffnn__Network   *message,
                      ProtobufCBuffer     *buffer);
Ffnn__Network *
       ffnn__network__unpack
                     (ProtobufCAllocator  *allocator,
                      size_t               len,
                      const uint8_t       *data);
void   ffnn__network__free_unpacked
                     (Ffnn__Network *message,
                      ProtobufCAllocator *allocator);
/* --- per-message closures --- */

typedef void (*Ffnn__Weight_Closure)
                 (const Ffnn__Weight *message,
                  void *closure_data);
typedef void (*Ffnn__Bias_Closure)
                 (const Ffnn__Bias *message,
                  void *closure_data);
typedef void (*Ffnn__Network_Closure)
                 (const Ffnn__Network *message,
                  void *closure_data);

/* --- services --- */


/* --- descriptors --- */

extern const ProtobufCMessageDescriptor ffnn__weight__descriptor;
extern const ProtobufCMessageDescriptor ffnn__bias__descriptor;
extern const ProtobufCMessageDescriptor ffnn__network__descriptor;
extern const ProtobufCEnumDescriptor    ffnn__network__activation_type__descriptor;

PROTOBUF_C__END_DECLS


#endif  /* PROTOBUF_C_network_2eproto__INCLUDED */
