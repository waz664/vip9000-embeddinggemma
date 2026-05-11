#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <vip_lite.h>

#define MAX_IO 8
#define MAX_LINE 4096

typedef struct {
    vip_network network;
    vip_buffer input[MAX_IO];
    vip_buffer output[MAX_IO];
    vip_uint32_t input_count;
    vip_uint32_t output_count;
} runner_t;

static int read_file_exact(const char *path, void *dst, size_t size) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "open input failed: %s: %s\n", path, strerror(errno));
        return 1;
    }
    size_t got = fread(dst, 1, size, fp);
    fclose(fp);
    if (got != size) {
        fprintf(stderr, "input size mismatch: %s got=%zu expected=%zu\n", path, got, size);
        return 1;
    }
    return 0;
}

static int write_file_exact(const char *path, const void *src, size_t size) {
    FILE *fp = fopen(path, "wb");
    if (!fp) {
        fprintf(stderr, "open output failed: %s: %s\n", path, strerror(errno));
        return 1;
    }
    size_t wrote = fwrite(src, 1, size, fp);
    fclose(fp);
    if (wrote != size) {
        fprintf(stderr, "output size mismatch: %s wrote=%zu expected=%zu\n", path, wrote, size);
        return 1;
    }
    return 0;
}

static int init_runner(runner_t *runner, const char *network_path) {
    memset(runner, 0, sizeof(*runner));
    vip_status_e status = vip_init();
    if (status != VIP_SUCCESS) {
        fprintf(stderr, "vip_init failed: %d\n", status);
        return 1;
    }

    status = vip_create_network(network_path, 0, VIP_CREATE_NETWORK_FROM_FILE, &runner->network);
    if (status != VIP_SUCCESS) {
        fprintf(stderr, "vip_create_network failed: %d\n", status);
        return 1;
    }

    vip_query_network(runner->network, VIP_NETWORK_PROP_INPUT_COUNT, &runner->input_count);
    vip_query_network(runner->network, VIP_NETWORK_PROP_OUTPUT_COUNT, &runner->output_count);
    if (runner->input_count > MAX_IO || runner->output_count > MAX_IO) {
        fprintf(stderr, "too many IO tensors: inputs=%u outputs=%u max=%u\n",
                runner->input_count, runner->output_count, MAX_IO);
        return 1;
    }

    for (vip_uint32_t i = 0; i < runner->input_count; ++i) {
        vip_buffer_create_params_t param;
        memset(&param, 0, sizeof(param));
        param.memory_type = VIP_BUFFER_MEMORY_TYPE_DEFAULT;
        vip_query_input(runner->network, i, VIP_BUFFER_PROP_DATA_FORMAT, &param.data_format);
        vip_query_input(runner->network, i, VIP_BUFFER_PROP_NUM_OF_DIMENSION, &param.num_of_dims);
        vip_query_input(runner->network, i, VIP_BUFFER_PROP_SIZES_OF_DIMENSION, param.sizes);
        vip_query_input(runner->network, i, VIP_BUFFER_PROP_QUANT_FORMAT, &param.quant_format);
        status = vip_create_buffer(&param, 0, &runner->input[i]);
        if (status != VIP_SUCCESS) {
            fprintf(stderr, "vip_create_buffer input %u failed: %d\n", i, status);
            return 1;
        }
    }

    for (vip_uint32_t i = 0; i < runner->output_count; ++i) {
        vip_buffer_create_params_t param;
        memset(&param, 0, sizeof(param));
        param.memory_type = VIP_BUFFER_MEMORY_TYPE_DEFAULT;
        vip_query_output(runner->network, i, VIP_BUFFER_PROP_DATA_FORMAT, &param.data_format);
        vip_query_output(runner->network, i, VIP_BUFFER_PROP_NUM_OF_DIMENSION, &param.num_of_dims);
        vip_query_output(runner->network, i, VIP_BUFFER_PROP_SIZES_OF_DIMENSION, param.sizes);
        vip_query_output(runner->network, i, VIP_BUFFER_PROP_QUANT_FORMAT, &param.quant_format);
        status = vip_create_buffer(&param, 0, &runner->output[i]);
        if (status != VIP_SUCCESS) {
            fprintf(stderr, "vip_create_buffer output %u failed: %d\n", i, status);
            return 1;
        }
    }

    status = vip_prepare_network(runner->network);
    if (status != VIP_SUCCESS) {
        fprintf(stderr, "vip_prepare_network failed: %d\n", status);
        return 1;
    }
    for (vip_uint32_t i = 0; i < runner->input_count; ++i) {
        status = vip_set_input(runner->network, i, runner->input[i]);
        if (status != VIP_SUCCESS) {
            fprintf(stderr, "vip_set_input %u failed: %d\n", i, status);
            return 1;
        }
    }
    for (vip_uint32_t i = 0; i < runner->output_count; ++i) {
        status = vip_set_output(runner->network, i, runner->output[i]);
        if (status != VIP_SUCCESS) {
            fprintf(stderr, "vip_set_output %u failed: %d\n", i, status);
            return 1;
        }
    }
    return 0;
}

static int run_once(runner_t *runner, const char **input_paths, const char **output_paths) {
    for (vip_uint32_t i = 0; i < runner->input_count; ++i) {
        void *ptr = vip_map_buffer(runner->input[i]);
        vip_uint32_t size = vip_get_buffer_size(runner->input[i]);
        if (!ptr || read_file_exact(input_paths[i], ptr, size)) {
            return 1;
        }
        vip_unmap_buffer(runner->input[i]);
        if (vip_flush_buffer(runner->input[i], VIP_BUFFER_OPER_TYPE_FLUSH) != VIP_SUCCESS) {
            fprintf(stderr, "vip_flush_buffer input %u failed\n", i);
            return 1;
        }
    }

    vip_status_e status = vip_run_network(runner->network);
    if (status != VIP_SUCCESS) {
        fprintf(stderr, "vip_run_network failed: %d\n", status);
        return 1;
    }

    for (vip_uint32_t i = 0; i < runner->output_count; ++i) {
        if (vip_flush_buffer(runner->output[i], VIP_BUFFER_OPER_TYPE_INVALIDATE) != VIP_SUCCESS) {
            fprintf(stderr, "vip_flush_buffer output %u failed\n", i);
            return 1;
        }
        void *ptr = vip_map_buffer(runner->output[i]);
        vip_uint32_t size = vip_get_buffer_size(runner->output[i]);
        if (!ptr || write_file_exact(output_paths[i], ptr, size)) {
            return 1;
        }
        vip_unmap_buffer(runner->output[i]);
    }
    return 0;
}

static void destroy_runner(runner_t *runner) {
    if (runner->network) {
        vip_finish_network(runner->network);
    }
    for (vip_uint32_t i = 0; i < runner->input_count; ++i) {
        if (runner->input[i]) {
            vip_destroy_buffer(runner->input[i]);
        }
    }
    for (vip_uint32_t i = 0; i < runner->output_count; ++i) {
        if (runner->output[i]) {
            vip_destroy_buffer(runner->output[i]);
        }
    }
    if (runner->network) {
        vip_destroy_network(runner->network);
    }
    vip_destroy();
}

static int serve_loop(runner_t *runner) {
    char line[MAX_LINE];
    const char *inputs[MAX_IO];
    const char *outputs[MAX_IO];
    char *tokens[MAX_IO * 2];

    printf("ready inputs=%u outputs=%u\n", runner->input_count, runner->output_count);
    fflush(stdout);
    while (fgets(line, sizeof(line), stdin)) {
        size_t n = 0;
        char *save = NULL;
        char *token = strtok_r(line, " \t\r\n", &save);
        while (token && n < MAX_IO * 2) {
            tokens[n++] = token;
            token = strtok_r(NULL, " \t\r\n", &save);
        }
        if (n == 1 && strcmp(tokens[0], "quit") == 0) {
            break;
        }
        if (n != runner->input_count + runner->output_count) {
            printf("error expected_paths=%u got=%zu\n", runner->input_count + runner->output_count, n);
            fflush(stdout);
            continue;
        }
        for (vip_uint32_t i = 0; i < runner->input_count; ++i) {
            inputs[i] = tokens[i];
        }
        for (vip_uint32_t i = 0; i < runner->output_count; ++i) {
            outputs[i] = tokens[runner->input_count + i];
        }
        int rc = run_once(runner, inputs, outputs);
        printf("%s\n", rc ? "error" : "ok");
        fflush(stdout);
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr,
                "usage:\n"
                "  %s --serve network.nb\n"
                "  %s network.nb input0.dat [inputN.dat ...] output0.dat [outputN.dat ...]\n",
                argv[0], argv[0]);
        return 2;
    }

    int serve = strcmp(argv[1], "--serve") == 0;
    const char *network_path = serve ? argv[2] : argv[1];
    runner_t runner;
    if (init_runner(&runner, network_path)) {
        destroy_runner(&runner);
        return 1;
    }

    int rc = 0;
    if (serve) {
        rc = serve_loop(&runner);
    } else {
        if ((unsigned)(argc - 2) != runner.input_count + runner.output_count) {
            fprintf(stderr, "expected %u IO paths, got %d\n", runner.input_count + runner.output_count, argc - 2);
            rc = 2;
        } else {
            const char *inputs[MAX_IO];
            const char *outputs[MAX_IO];
            for (vip_uint32_t i = 0; i < runner.input_count; ++i) {
                inputs[i] = argv[2 + i];
            }
            for (vip_uint32_t i = 0; i < runner.output_count; ++i) {
                outputs[i] = argv[2 + runner.input_count + i];
            }
            rc = run_once(&runner, inputs, outputs);
        }
    }
    destroy_runner(&runner);
    return rc;
}
