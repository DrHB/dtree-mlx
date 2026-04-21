#ifndef DTREE_MLX_METAL_SHIM_H
#define DTREE_MLX_METAL_SHIM_H

#include <stddef.h>
#include <stdint.h>

typedef uint32_t dt_metal_status;
enum {
    DT_METAL_STATUS_OK = 0,
    DT_METAL_STATUS_OUT_OF_MEMORY = 1,
    DT_METAL_STATUS_NO_DEVICE = 2,
    DT_METAL_STATUS_STRING_CREATION_FAILED = 3,
    DT_METAL_STATUS_SHADER_COMPILATION_FAILED = 4,
    DT_METAL_STATUS_FUNCTION_LOOKUP_FAILED = 5,
    DT_METAL_STATUS_PIPELINE_CREATION_FAILED = 6,
    DT_METAL_STATUS_COMMAND_QUEUE_CREATION_FAILED = 7,
    DT_METAL_STATUS_COMMAND_BUFFER_CREATION_FAILED = 8,
    DT_METAL_STATUS_COMPUTE_ENCODER_CREATION_FAILED = 9,
    DT_METAL_STATUS_COMMAND_EXECUTION_FAILED = 10,
    DT_METAL_STATUS_BUFFER_CREATION_FAILED = 11,
    DT_METAL_STATUS_BUFFER_MAP_FAILED = 12,
};

typedef struct dt_metal_session dt_metal_session;
typedef struct dt_metal_pipeline dt_metal_pipeline;
typedef struct dt_metal_buffer dt_metal_buffer;

typedef struct {
    dt_metal_buffer *handle;
    float *floats;
} dt_metal_buffer_info;

typedef struct {
    dt_metal_pipeline *handle;
    size_t thread_execution_width;
    size_t max_total_threads_per_threadgroup;
} dt_metal_pipeline_info;

dt_metal_status dt_metal_copy_executable_dir(char **out_dir, char **error_out);

dt_metal_status dt_metal_session_create(
    const char *source_utf8,
    dt_metal_session **out_session,
    char **error_out
);
void dt_metal_session_destroy(dt_metal_session *session);
dt_metal_status dt_metal_session_copy_device_name(
    dt_metal_session *session,
    char **out_name,
    char **error_out
);

dt_metal_status dt_metal_buffer_create(
    dt_metal_session *session,
    size_t element_count,
    dt_metal_buffer_info *out_info,
    char **error_out
);
void dt_metal_buffer_destroy(dt_metal_buffer *buffer);

dt_metal_status dt_metal_pipeline_create(
    dt_metal_session *session,
    const char *function_name,
    dt_metal_pipeline_info *out_info,
    char **error_out
);
void dt_metal_pipeline_destroy(dt_metal_pipeline *pipeline);

dt_metal_status dt_metal_dispatch_add_one(
    dt_metal_session *session,
    dt_metal_pipeline *pipeline,
    dt_metal_buffer *buffer,
    size_t element_count,
    size_t threads_per_group_width,
    char **error_out
);

dt_metal_status dt_metal_dispatch_dense_matvec(
    dt_metal_session *session,
    dt_metal_pipeline *pipeline,
    dt_metal_buffer *matrix,
    dt_metal_buffer *vector,
    dt_metal_buffer *output,
    size_t threadgroup_count,
    size_t threadgroup_width,
    uint32_t rows,
    uint32_t cols,
    char **error_out
);

void dt_metal_string_free(char *value);

#endif
