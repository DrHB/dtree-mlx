#import "shim.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <mach-o/dyld.h>
#import <stdlib.h>
#import <string.h>

struct dt_metal_session {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLLibrary> library;
};

struct dt_metal_pipeline {
    id<MTLFunction> function;
    id<MTLComputePipelineState> pipeline;
};

struct dt_metal_buffer {
    id<MTLBuffer> buffer;
    float *floats;
};

static void dt_metal_clear_error(char **error_out) {
    if (error_out != NULL) {
        *error_out = NULL;
    }
}

static char *dt_metal_copy_c_string(const char *value) {
    if (value == NULL) {
        return NULL;
    }

    const size_t len = strlen(value) + 1;
    char *copy = malloc(len);
    if (copy == NULL) {
        return NULL;
    }

    memcpy(copy, value, len);
    return copy;
}

static void dt_metal_set_error_c_string(char **error_out, const char *message) {
    if (error_out == NULL) {
        return;
    }

    *error_out = dt_metal_copy_c_string(message != NULL ? message : "unknown metal error");
}

static void dt_metal_set_error_string(char **error_out, NSString *message) {
    dt_metal_set_error_c_string(error_out, message != nil ? message.UTF8String : NULL);
}

static void dt_metal_set_error_nserror(char **error_out, NSError *error) {
    if (error != nil) {
        dt_metal_set_error_string(error_out, error.localizedDescription);
    } else {
        dt_metal_set_error_c_string(error_out, "unknown metal error");
    }
}

void dt_metal_string_free(char *value) {
    free(value);
}

dt_metal_status dt_metal_copy_executable_dir(char **out_dir, char **error_out) {
    @autoreleasepool {
        if (out_dir != NULL) {
            *out_dir = NULL;
        }
        dt_metal_clear_error(error_out);

        uint32_t buffer_len = 0;
        _NSGetExecutablePath(NULL, &buffer_len);
        if (buffer_len == 0) {
            dt_metal_set_error_c_string(error_out, "unable to determine executable path");
            return DT_METAL_STATUS_STRING_CREATION_FAILED;
        }

        char *buffer = malloc(buffer_len);
        if (buffer == NULL) {
            dt_metal_set_error_c_string(error_out, "out of memory");
            return DT_METAL_STATUS_OUT_OF_MEMORY;
        }

        if (_NSGetExecutablePath(buffer, &buffer_len) != 0) {
            free(buffer);
            dt_metal_set_error_c_string(error_out, "unable to determine executable path");
            return DT_METAL_STATUS_STRING_CREATION_FAILED;
        }

        char *resolved = realpath(buffer, NULL);
        const char *path = resolved != NULL ? resolved : buffer;

        NSString *full_path = [[NSString alloc] initWithUTF8String:path];
        free(resolved);
        free(buffer);
        if (full_path == nil) {
            dt_metal_set_error_c_string(error_out, "unable to determine executable path");
            return DT_METAL_STATUS_STRING_CREATION_FAILED;
        }

        NSString *dir = [full_path stringByDeletingLastPathComponent];
        char *copy = dt_metal_copy_c_string(dir.UTF8String);
        [full_path release];

        if (copy == NULL) {
            dt_metal_set_error_c_string(error_out, "out of memory");
            return DT_METAL_STATUS_OUT_OF_MEMORY;
        }

        if (out_dir != NULL) {
            *out_dir = copy;
        }
        return DT_METAL_STATUS_OK;
    }
}

dt_metal_status dt_metal_session_create(
    const char *source_utf8,
    dt_metal_session **out_session,
    char **error_out
) {
    @autoreleasepool {
        if (out_session != NULL) {
            *out_session = NULL;
        }
        dt_metal_clear_error(error_out);

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            return DT_METAL_STATUS_NO_DEVICE;
        }
        [device retain];

        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (queue == nil) {
            [device release];
            return DT_METAL_STATUS_COMMAND_QUEUE_CREATION_FAILED;
        }

        NSString *source = [[NSString alloc] initWithUTF8String:source_utf8];
        if (source == nil) {
            [queue release];
            [device release];
            return DT_METAL_STATUS_STRING_CREATION_FAILED;
        }

        NSError *compile_error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&compile_error];
        [source release];
        if (library == nil) {
            dt_metal_set_error_nserror(error_out, compile_error);
            [queue release];
            [device release];
            return DT_METAL_STATUS_SHADER_COMPILATION_FAILED;
        }

        dt_metal_session *session = calloc(1, sizeof(*session));
        if (session == NULL) {
            dt_metal_set_error_c_string(error_out, "out of memory");
            [library release];
            [queue release];
            [device release];
            return DT_METAL_STATUS_OUT_OF_MEMORY;
        }

        session->device = device;
        session->queue = queue;
        session->library = library;
        if (out_session != NULL) {
            *out_session = session;
        }
        return DT_METAL_STATUS_OK;
    }
}

void dt_metal_session_destroy(dt_metal_session *session) {
    if (session == NULL) {
        return;
    }

    [session->library release];
    [session->queue release];
    [session->device release];
    free(session);
}

dt_metal_status dt_metal_session_copy_device_name(
    dt_metal_session *session,
    char **out_name,
    char **error_out
) {
    @autoreleasepool {
        if (out_name != NULL) {
            *out_name = NULL;
        }
        dt_metal_clear_error(error_out);

        NSString *name = session != NULL ? session->device.name : nil;
        if (name == nil) {
            return DT_METAL_STATUS_STRING_CREATION_FAILED;
        }

        char *copy = dt_metal_copy_c_string(name.UTF8String);
        if (copy == NULL) {
            dt_metal_set_error_c_string(error_out, "out of memory");
            return DT_METAL_STATUS_OUT_OF_MEMORY;
        }

        if (out_name != NULL) {
            *out_name = copy;
        }
        return DT_METAL_STATUS_OK;
    }
}

dt_metal_status dt_metal_buffer_create(
    dt_metal_session *session,
    size_t element_count,
    dt_metal_buffer_info *out_info,
    char **error_out
) {
    @autoreleasepool {
        if (out_info != NULL) {
            out_info->handle = NULL;
            out_info->contents = NULL;
            out_info->floats = NULL;
        }
        dt_metal_clear_error(error_out);

        if (element_count > SIZE_MAX / sizeof(float)) {
            dt_metal_set_error_c_string(error_out, "buffer size overflow");
            return DT_METAL_STATUS_BUFFER_CREATION_FAILED;
        }

        const size_t byte_len = element_count * sizeof(float);
        id<MTLBuffer> buffer = [session->device newBufferWithLength:byte_len options:MTLResourceStorageModeShared];
        if (buffer == nil) {
            return DT_METAL_STATUS_BUFFER_CREATION_FAILED;
        }

        void *contents = [buffer contents];
        if (contents == NULL) {
            [buffer release];
            return DT_METAL_STATUS_BUFFER_MAP_FAILED;
        }

        dt_metal_buffer *wrapped = calloc(1, sizeof(*wrapped));
        if (wrapped == NULL) {
            dt_metal_set_error_c_string(error_out, "out of memory");
            [buffer release];
            return DT_METAL_STATUS_OUT_OF_MEMORY;
        }

        wrapped->buffer = buffer;
        wrapped->floats = (float *)contents;
        if (out_info != NULL) {
            out_info->handle = wrapped;
            out_info->contents = contents;
            out_info->floats = wrapped->floats;
        }
        return DT_METAL_STATUS_OK;
    }
}

dt_metal_status dt_metal_buffer_wrap_bytes_no_copy(
    dt_metal_session *session,
    const void *bytes,
    size_t byte_length,
    dt_metal_buffer **out_buffer,
    char **error_out
) {
    @autoreleasepool {
        if (out_buffer != NULL) {
            *out_buffer = NULL;
        }
        dt_metal_clear_error(error_out);

        id<MTLBuffer> buffer = [session->device newBufferWithBytesNoCopy:(void *)bytes
                                                                  length:byte_length
                                                                 options:MTLResourceStorageModeShared
                                                             deallocator:nil];
        if (buffer == nil) {
            return DT_METAL_STATUS_BUFFER_CREATION_FAILED;
        }

        dt_metal_buffer *wrapped = calloc(1, sizeof(*wrapped));
        if (wrapped == NULL) {
            dt_metal_set_error_c_string(error_out, "out of memory");
            [buffer release];
            return DT_METAL_STATUS_OUT_OF_MEMORY;
        }

        wrapped->buffer = buffer;
        wrapped->floats = NULL;
        if (out_buffer != NULL) {
            *out_buffer = wrapped;
        }
        return DT_METAL_STATUS_OK;
    }
}

void dt_metal_buffer_destroy(dt_metal_buffer *buffer) {
    if (buffer == NULL) {
        return;
    }

    [buffer->buffer release];
    free(buffer);
}

dt_metal_status dt_metal_pipeline_create(
    dt_metal_session *session,
    const char *function_name,
    dt_metal_pipeline_info *out_info,
    char **error_out
) {
    @autoreleasepool {
        if (out_info != NULL) {
            out_info->handle = NULL;
            out_info->thread_execution_width = 0;
            out_info->max_total_threads_per_threadgroup = 0;
        }
        dt_metal_clear_error(error_out);

        NSString *name = [[NSString alloc] initWithUTF8String:function_name];
        if (name == nil) {
            return DT_METAL_STATUS_STRING_CREATION_FAILED;
        }

        id<MTLFunction> function = [session->library newFunctionWithName:name];
        [name release];
        if (function == nil) {
            return DT_METAL_STATUS_FUNCTION_LOOKUP_FAILED;
        }

        NSError *pipeline_error = nil;
        id<MTLComputePipelineState> pipeline = [session->device newComputePipelineStateWithFunction:function error:&pipeline_error];
        if (pipeline == nil) {
            dt_metal_set_error_nserror(error_out, pipeline_error);
            [function release];
            return DT_METAL_STATUS_PIPELINE_CREATION_FAILED;
        }

        dt_metal_pipeline *wrapped = calloc(1, sizeof(*wrapped));
        if (wrapped == NULL) {
            dt_metal_set_error_c_string(error_out, "out of memory");
            [pipeline release];
            [function release];
            return DT_METAL_STATUS_OUT_OF_MEMORY;
        }

        wrapped->function = function;
        wrapped->pipeline = pipeline;
        if (out_info != NULL) {
            out_info->handle = wrapped;
            out_info->thread_execution_width = pipeline.threadExecutionWidth;
            out_info->max_total_threads_per_threadgroup = pipeline.maxTotalThreadsPerThreadgroup;
        }
        return DT_METAL_STATUS_OK;
    }
}

void dt_metal_pipeline_destroy(dt_metal_pipeline *pipeline) {
    if (pipeline == NULL) {
        return;
    }

    [pipeline->pipeline release];
    [pipeline->function release];
    free(pipeline);
}

static dt_metal_status dt_metal_finish_command(
    id<MTLCommandBuffer> command_buffer,
    char **error_out
) {
    [command_buffer commit];
    [command_buffer waitUntilCompleted];

    if (command_buffer.error != nil) {
        dt_metal_set_error_nserror(error_out, command_buffer.error);
        return DT_METAL_STATUS_COMMAND_EXECUTION_FAILED;
    }

    return DT_METAL_STATUS_OK;
}

dt_metal_status dt_metal_dispatch_add_one(
    dt_metal_session *session,
    dt_metal_pipeline *pipeline,
    dt_metal_buffer *buffer,
    size_t element_count,
    size_t threads_per_group_width,
    char **error_out
) {
    @autoreleasepool {
        dt_metal_clear_error(error_out);

        id<MTLCommandBuffer> command_buffer = [session->queue commandBuffer];
        if (command_buffer == nil) {
            return DT_METAL_STATUS_COMMAND_BUFFER_CREATION_FAILED;
        }

        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (encoder == nil) {
            return DT_METAL_STATUS_COMPUTE_ENCODER_CREATION_FAILED;
        }

        [encoder setComputePipelineState:pipeline->pipeline];
        [encoder setBuffer:buffer->buffer offset:0 atIndex:0];
        [encoder dispatchThreads:MTLSizeMake(element_count, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(threads_per_group_width, 1, 1)];
        [encoder endEncoding];

        return dt_metal_finish_command(command_buffer, error_out);
    }
}

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
) {
    @autoreleasepool {
        dt_metal_clear_error(error_out);

        id<MTLCommandBuffer> command_buffer = [session->queue commandBuffer];
        if (command_buffer == nil) {
            return DT_METAL_STATUS_COMMAND_BUFFER_CREATION_FAILED;
        }

        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (encoder == nil) {
            return DT_METAL_STATUS_COMPUTE_ENCODER_CREATION_FAILED;
        }

        [encoder setComputePipelineState:pipeline->pipeline];
        [encoder setBuffer:matrix->buffer offset:0 atIndex:0];
        [encoder setBuffer:vector->buffer offset:0 atIndex:1];
        [encoder setBuffer:output->buffer offset:0 atIndex:2];
        [encoder setBytes:&cols length:sizeof(cols) atIndex:3];
        [encoder setBytes:&rows length:sizeof(rows) atIndex:4];
        [encoder dispatchThreadgroups:MTLSizeMake(threadgroup_count, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(threadgroup_width, 1, 1)];
        [encoder endEncoding];

        return dt_metal_finish_command(command_buffer, error_out);
    }
}

static dt_metal_status dt_metal_dispatch_quant_matvec(
    dt_metal_session *session,
    dt_metal_pipeline *pipeline,
    dt_metal_buffer *weights,
    dt_metal_buffer *vector,
    dt_metal_buffer *output,
    size_t threadgroup_count,
    size_t threadgroup_width,
    uint32_t row_stride_bytes,
    uint32_t rows,
    uint32_t cols,
    char **error_out
) {
    @autoreleasepool {
        dt_metal_clear_error(error_out);

        id<MTLCommandBuffer> command_buffer = [session->queue commandBuffer];
        if (command_buffer == nil) {
            return DT_METAL_STATUS_COMMAND_BUFFER_CREATION_FAILED;
        }

        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (encoder == nil) {
            return DT_METAL_STATUS_COMPUTE_ENCODER_CREATION_FAILED;
        }

        [encoder setComputePipelineState:pipeline->pipeline];
        [encoder setBuffer:weights->buffer offset:0 atIndex:0];
        [encoder setBuffer:vector->buffer offset:0 atIndex:1];
        [encoder setBuffer:output->buffer offset:0 atIndex:2];
        [encoder setBytes:&row_stride_bytes length:sizeof(row_stride_bytes) atIndex:3];
        [encoder setBytes:&cols length:sizeof(cols) atIndex:4];
        [encoder setBytes:&rows length:sizeof(rows) atIndex:5];
        [encoder dispatchThreadgroups:MTLSizeMake(threadgroup_count, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(threadgroup_width, 1, 1)];
        [encoder endEncoding];

        return dt_metal_finish_command(command_buffer, error_out);
    }
}

dt_metal_status dt_metal_dispatch_q4_k_matvec(
    dt_metal_session *session,
    dt_metal_pipeline *pipeline,
    dt_metal_buffer *weights,
    dt_metal_buffer *vector,
    dt_metal_buffer *output,
    size_t threadgroup_count,
    size_t threadgroup_width,
    uint32_t row_stride_bytes,
    uint32_t rows,
    uint32_t cols,
    char **error_out
) {
    return dt_metal_dispatch_quant_matvec(
        session,
        pipeline,
        weights,
        vector,
        output,
        threadgroup_count,
        threadgroup_width,
        row_stride_bytes,
        rows,
        cols,
        error_out
    );
}

dt_metal_status dt_metal_dispatch_q6_k_matvec(
    dt_metal_session *session,
    dt_metal_pipeline *pipeline,
    dt_metal_buffer *weights,
    dt_metal_buffer *vector,
    dt_metal_buffer *output,
    size_t threadgroup_count,
    size_t threadgroup_width,
    uint32_t row_stride_bytes,
    uint32_t rows,
    uint32_t cols,
    char **error_out
) {
    return dt_metal_dispatch_quant_matvec(
        session,
        pipeline,
        weights,
        vector,
        output,
        threadgroup_count,
        threadgroup_width,
        row_stride_bytes,
        rows,
        cols,
        error_out
    );
}
