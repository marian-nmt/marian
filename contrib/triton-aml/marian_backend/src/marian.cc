#include <algorithm>
#include "marian.h"
#include "triton/backend/backend_common.h"

namespace triton { namespace backend { namespace marian {

#define GUARDED_RESPOND_IF_ERROR(RESPONSES, IDX, X)                             \
    do {                                                                        \
        if ((RESPONSES)[IDX] != nullptr) {                                      \
            TRITONSERVER_Error* err__ = (X);                                    \
            if (err__ != nullptr) {                                             \
                LOG_IF_ERROR(                                                   \
                    TRITONBACKEND_ResponseSend(                                 \
                        (RESPONSES)[IDX], TRITONSERVER_RESPONSE_COMPLETE_FINAL, \
                        err__),                                                 \
                    "failed to send error response");                           \
                (RESPONSES)[IDX] = nullptr;                                     \
                TRITONSERVER_ErrorDelete(err__);                                \
            }                                                                   \
        }                                                                       \
    } while (false)

//
// ModelState
//
// State associated with a model that is using this backend. An object
// of this class is created and associated with each
// TRITONBACKEND_Model.
//
class ModelState {
public:
    static TRITONSERVER_Error* Create(
        TRITONBACKEND_Model* triton_model, ModelState** state);

    TRITONSERVER_Error* SetMarianConfigPath();

    // Get the handle to the TRITONBACKEND model.
    TRITONBACKEND_Model* TritonModel() { return triton_model_; }

    // Get the name of the model.
    const std::string& Name() const { return name_; }

    // Get the Marian config path of the model.
    const std::string& MarianConfigPath() const { return marian_config_path_; }

private:
    ModelState(
        TRITONBACKEND_Model* triton_model, const char* name,
        common::TritonJson::Value&& model_config);

    TRITONBACKEND_Model* triton_model_;
    const std::string name_;
    common::TritonJson::Value model_config_;
    std::string marian_config_path_;
};

TRITONSERVER_Error*
ModelState::Create(TRITONBACKEND_Model* triton_model, ModelState** state)
{
    TRITONSERVER_Message* config_message;
    RETURN_IF_ERROR(TRITONBACKEND_ModelConfig(
        triton_model, 1 /* config_version */, &config_message));

    // Get the model configuration as a json string from
    // config_message, parse it with the TritonJson.
    const char* buffer;
    size_t byte_size;
    RETURN_IF_ERROR(
        TRITONSERVER_MessageSerializeToJson(config_message, &buffer, &byte_size));

    common::TritonJson::Value model_config;
    TRITONSERVER_Error* err = model_config.Parse(buffer, byte_size);
    RETURN_IF_ERROR(TRITONSERVER_MessageDelete(config_message));
    RETURN_IF_ERROR(err);

    const char* model_name;
    RETURN_IF_ERROR(TRITONBACKEND_ModelName(triton_model, &model_name));

    *state = new ModelState(
        triton_model, model_name, std::move(model_config));

    return nullptr;  // success
}

ModelState::ModelState(
    TRITONBACKEND_Model* triton_model, const char* name,
    common::TritonJson::Value&& model_config)
    : triton_model_(triton_model), name_(name),
      model_config_(std::move(model_config))
{
}

TRITONSERVER_Error*
ModelState::SetMarianConfigPath()
{
    common::TritonJson::WriteBuffer buffer;
    RETURN_IF_ERROR(model_config_.PrettyWrite(&buffer));
    LOG_MESSAGE(
        TRITONSERVER_LOG_INFO,
        (std::string("model configuration:\n") + buffer.Contents()).c_str());

    std::string config_filepath_str;
    common::TritonJson::Value parameters;
    if (model_config_.Find("parameters", &parameters)) {
        common::TritonJson::Value config_filepath;
        if (parameters.Find("config_filepath", &config_filepath)) {
            RETURN_IF_ERROR(config_filepath.MemberAsString(
                "string_value", &config_filepath_str)
            );
            LOG_MESSAGE(
                TRITONSERVER_LOG_INFO,
                (std::string("model config path is set to : ") + config_filepath_str)
                .c_str()
            );
        }
    }

    // Set the Marian config path.
    std::string config_path("/var/azureml-app/");
    config_path.append(std::getenv("AZUREML_MODEL_DIR"));
    config_path.append(config_filepath_str);
    marian_config_path_ = config_path;

    return nullptr;  // success
}

//
// ModelInstanceState
//
// State associated with a model instance. An object of this class is
// created and associated with each TRITONBACKEND_ModelInstance.
//
class ModelInstanceState {
public:
    static TRITONSERVER_Error* Create(
        TRITONBACKEND_ModelInstance* triton_model_instance,
        void* marian, ModelInstanceState **state);

    // Get the handle to the TRITONBACKEND model instance.
    TRITONBACKEND_ModelInstance* TritonModelInstance()
    {
        return triton_model_instance_;
    }

    // Get the name, kind, device ID and marian instance of the instance.
    const std::string& Name() const { return name_; }
    TRITONSERVER_InstanceGroupKind Kind() const { return kind_; }
    int32_t DeviceId() const { return device_id_; }
    void* Marian() const { return marian_; }

private:
    ModelInstanceState(
        TRITONBACKEND_ModelInstance* triton_model_instance,
        void* marian, const char* name,
        const TRITONSERVER_InstanceGroupKind kind, const int32_t device_id);

    TRITONBACKEND_ModelInstance* triton_model_instance_;
    void* marian_;
    const std::string name_;
    const TRITONSERVER_InstanceGroupKind kind_;
    const int32_t device_id_;
};

TRITONSERVER_Error*
ModelInstanceState::Create(
    TRITONBACKEND_ModelInstance* triton_model_instance,
    void* marian, ModelInstanceState** state)
{
    const char* instance_name;
    RETURN_IF_ERROR(
        TRITONBACKEND_ModelInstanceName(triton_model_instance, &instance_name));

    TRITONSERVER_InstanceGroupKind instance_kind;
    RETURN_IF_ERROR(
        TRITONBACKEND_ModelInstanceKind(triton_model_instance, &instance_kind));

    int32_t instance_id;
    RETURN_IF_ERROR(
        TRITONBACKEND_ModelInstanceDeviceId(triton_model_instance, &instance_id));

    *state = new ModelInstanceState(
        triton_model_instance, marian, instance_name,
        instance_kind, instance_id);

    return nullptr;  // success
}

ModelInstanceState::ModelInstanceState(
    TRITONBACKEND_ModelInstance* triton_model_instance,
    void* marian, const char* name,
    const TRITONSERVER_InstanceGroupKind kind, const int32_t device_id)
    : triton_model_instance_(triton_model_instance), marian_(marian),
      name_(name), kind_(kind), device_id_(device_id)
{
}

/////////////

extern "C" {

void
handler(int sig) {
    void* array[30];

    size_t size = backtrace(array, 30);

    fprintf(stderr, "Error: signal %d, Exception info:\n", sig);
    backtrace_symbols_fd(array, size, STDERR_FILENO);
}

TRITONSERVER_Error*
TRITONBACKEND_ModelInitialize(TRITONBACKEND_Model* model)
{
    ModelState* model_state;
    RETURN_IF_ERROR(ModelState::Create(model, &model_state));
    RETURN_IF_ERROR(model_state->SetMarianConfigPath());
    RETURN_IF_ERROR(
        TRITONBACKEND_ModelSetState(model, reinterpret_cast<void*>(model_state))
    );

    signal(SIGSEGV, handler);
    signal(SIGABRT, handler);

    return nullptr; // success
}

TRITONSERVER_Error*
TRITONBACKEND_ModelFinalize(TRITONBACKEND_Model* model)
{
    void* vstate;
    RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vstate));
    ModelState* model_state = reinterpret_cast<ModelState*>(vstate);

    LOG_MESSAGE(
        TRITONSERVER_LOG_INFO, "TRITONBACKEND_ModelFinalize: delete model state");

    delete model_state;

    return nullptr;  // success
}

TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceInitialize(TRITONBACKEND_ModelInstance* instance)
{
    TRITONBACKEND_Model* model;
    RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceModel(instance, &model));

    void* vmodelstate;
    RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vmodelstate));
    ModelState* model_state = reinterpret_cast<ModelState*>(vmodelstate);

    std::string marian_config_path = model_state->MarianConfigPath();

    int32_t device;
    RETURN_IF_ERROR(
        TRITONBACKEND_ModelInstanceDeviceId(instance, &device));

    void* marian_instance = init(const_cast<char*>(marian_config_path.c_str()), device);

    ModelInstanceState* instance_state;
    RETURN_IF_ERROR(
        ModelInstanceState::Create(instance, marian_instance, &instance_state));
    RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceSetState(
        instance, reinterpret_cast<void*>(instance_state)));

    return nullptr;  // success
}

TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceFinalize(TRITONBACKEND_ModelInstance* instance)
{
    void* vstate;
    RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(instance, &vstate));
    ModelInstanceState* instance_state =
        reinterpret_cast<ModelInstanceState*>(vstate);

    LOG_MESSAGE(
        TRITONSERVER_LOG_INFO,
        "TRITONBACKEND_ModelInstanceFinalize: delete instance state");

    delete instance_state;

    return nullptr;  // success
}

TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceExecute(
    TRITONBACKEND_ModelInstance* instance, TRITONBACKEND_Request** requests,
    const uint32_t request_count)
{
    LOG_MESSAGE(
        TRITONSERVER_LOG_INFO,
        ("Marian model instance executing " + std::to_string(request_count) +
         " requests").c_str()
    );

    // 'responses' is initialized with the response objects below and
    // if/when an error response is sent the corresponding entry in
    // 'responses' is set to nullptr to indicate that that response has
    // already been sent.
    std::vector<TRITONBACKEND_Response*> responses;
    responses.reserve(request_count);

    // Create a single response object for each request. If something
    // goes wrong when attempting to create the response objects just
    // fail all of the requests by returning an error.
    for (uint32_t r = 0; r < request_count; ++r) {
        TRITONBACKEND_Request* request = requests[r];

        TRITONBACKEND_Response* response;
        RETURN_IF_ERROR(TRITONBACKEND_ResponseNew(&response, request));
        responses.push_back(response);
    }

    // We will execute all the requests at the same time, and so there
    // will be a single compute-start / compute-end time-range.
    uint64_t total_batch_size = 0;
    uint64_t exec_start_ns = 0;
    SET_TIMESTAMP(exec_start_ns);

    std::vector<TRITONBACKEND_Input*> request_input;
    std::vector<int> request_batch_size;
    std::string input_strings;

    // Create a single response object for each request. If something
    // goes wrong when attempting to create the response objects just
    // fail all of the requests by returning an error.
    for (uint32_t r = 0; r < request_count; ++r) {
        TRITONBACKEND_Request* request = requests[r];

        const char* input_name;
        GUARDED_RESPOND_IF_ERROR(
            responses, r,
            TRITONBACKEND_RequestInputName(request, 0 /* index */, &input_name)
        );

        TRITONBACKEND_Input* input = nullptr;
        GUARDED_RESPOND_IF_ERROR(
            responses, r,
            TRITONBACKEND_RequestInput(request, input_name, &input)
        );
        request_input.push_back(input);

        // If an error response was sent while getting the input name
        // or input then display an error message and move on
        // to next request.
        if (responses[r] == nullptr) {
            LOG_MESSAGE(
                TRITONSERVER_LOG_ERROR,
                (std::string("request ") + std::to_string(r) +
                 ": failed to read input or requested output name, error response sent")
                 .c_str()
            );
            continue;
        }

        // Get input buffer count.
        uint32_t input_buffer_count;
        GUARDED_RESPOND_IF_ERROR(
            responses, r,
            TRITONBACKEND_InputProperties(
                input, nullptr /* input_name */, nullptr, nullptr,
                nullptr, nullptr, &input_buffer_count
            )
        );
        if (responses[r] == nullptr) {
            LOG_MESSAGE(
                TRITONSERVER_LOG_ERROR,
                (std::string("request ") + std::to_string(r) +
                 ": failed to read input properties, error response sent")
                 .c_str()
            );
            continue;
        }

        // Compose all the requests input to make a batch request,
        // record the sentences count of each request for further process.
        std::vector<char> content_buffer;
        for (uint32_t b = 0; b < input_buffer_count; ++b) {
            const void* input_buffer = nullptr;
            uint64_t buffer_byte_size = 0;
            TRITONSERVER_MemoryType input_memory_type = TRITONSERVER_MEMORY_CPU;
            int64_t input_memory_type_id = 0;
            GUARDED_RESPOND_IF_ERROR(
                responses, r,
                TRITONBACKEND_InputBuffer(
                    input, b, &input_buffer, &buffer_byte_size,
                    &input_memory_type, &input_memory_type_id
                )
            );
            if ((responses[r] == nullptr) ||
                (input_memory_type == TRITONSERVER_MEMORY_GPU)) {
                GUARDED_RESPOND_IF_ERROR(
                    responses, r,
                    TRITONSERVER_ErrorNew(
                        TRITONSERVER_ERROR_UNSUPPORTED,
                        "failed to get input buffer in CPU memory"
                    )
                );
            }
            content_buffer.insert(
                content_buffer.end(), reinterpret_cast<const char*>(input_buffer) + 4,
                reinterpret_cast<const char*>(input_buffer) + buffer_byte_size
            );
        }

        std::string s(content_buffer.begin(), content_buffer.end());
        int count = std::count(s.begin(), s.end(), '\n');
        request_batch_size.push_back(count + 1);
        content_buffer.clear();

        if (input_strings.empty()) {
            input_strings = s;
        } else {
            input_strings.append("\n");
            input_strings.append(s);
        }

        total_batch_size++;
    }

    // Operate on the entire batch of requests for improved performance.
    void* vstate;
    RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(instance, &vstate));
    ModelInstanceState* instance_state =
        reinterpret_cast<ModelInstanceState*>(vstate);
    void* marian = instance_state->Marian();
    char* result = translate(marian, const_cast<char*>(input_strings.c_str()));

    // Assign the results to the corresponding request.
    char* pos = result;
    for (uint32_t r = 0; r < request_count; ++r) {
        int batch_size = request_batch_size[r];
        uint64_t output_byte_size = 0;
        char* output_content = nullptr;

        // Find current output content.
        while (batch_size > 0) {
            char* p = strchr(pos, '\n');
            if (p != nullptr) {
                *p = '\0';
            }
            if (output_content == nullptr) {
                output_content = pos;
            } else {
                // Replace the null terminator of the prev sentence with new line char
                *(pos - 1) = '\n';
            }
            // Move to next output content.
            if (p != nullptr) {
                pos = p + 1;
            } else {
                // Break if there no left output content, even though batch_size > 0,
                // '\n' at the end may be processed by Marian.
                break;
            }
            batch_size--;
        }
        output_byte_size = strlen(output_content);

        TRITONBACKEND_Input* input = request_input[r];
        const char* input_name;
        TRITONSERVER_DataType input_datatype;
        const int64_t* input_shape;
        uint32_t input_dims_count;
        uint64_t input_byte_size;
        uint32_t input_buffer_count;
        GUARDED_RESPOND_IF_ERROR(
            responses, r,
            TRITONBACKEND_InputProperties(
                input, &input_name, &input_datatype, &input_shape,
                &input_dims_count, &input_byte_size, &input_buffer_count
            )
        );
        if (responses[r] == nullptr) {
            LOG_MESSAGE(
                TRITONSERVER_LOG_ERROR,
                (std::string("request ") + std::to_string(r) +
                 ": failed to read input properties, error response sent")
                 .c_str()
            );
            continue;
        }

        TRITONBACKEND_Request* request = requests[r];
        const char* requested_output_name = nullptr;
        GUARDED_RESPOND_IF_ERROR(
            responses, r,
            TRITONBACKEND_RequestOutputName(
                request, 0 /* index */, &requested_output_name
            )
        );

        // Create an output tensor in the response,
        // input and output have same datatype and shape...
        TRITONBACKEND_Response* response = responses[r];
        TRITONBACKEND_Output* output;
        GUARDED_RESPOND_IF_ERROR(
            responses, r,
            TRITONBACKEND_ResponseOutput(
                response, &output, requested_output_name, input_datatype,
                input_shape, input_dims_count
            )
        );

        // Get the output buffer. We request a buffer in CPU memory
        // but we have to handle any returned type. If we get back
        // a buffer in GPU memory we just fail the request.
        void* output_buffer;
        TRITONSERVER_MemoryType output_memory_type = TRITONSERVER_MEMORY_CPU;
        int64_t output_memory_type_id = 0;
        GUARDED_RESPOND_IF_ERROR(
            responses, r,
            TRITONBACKEND_OutputBuffer(
                output, &output_buffer, output_byte_size + 4,
                &output_memory_type, &output_memory_type_id
            )
        );
        if ((responses[r] == nullptr) ||
            (output_memory_type == TRITONSERVER_MEMORY_GPU)) {
            GUARDED_RESPOND_IF_ERROR(
                responses, r,
                TRITONSERVER_ErrorNew(
                    TRITONSERVER_ERROR_UNSUPPORTED,
                    "failed to create output buffer in CPU memory"
                )
            );
            LOG_MESSAGE(
                TRITONSERVER_LOG_ERROR,
                (std::string("request ") + std::to_string(r) +
                ": failed to create output buffer in CPU memory, error request sent")
                .c_str()
            );
            continue;
        }

        // Copy Marian result -> output.
        memcpy(output_buffer, reinterpret_cast<char*>(&output_byte_size), 4);
        memcpy(reinterpret_cast<char*>(output_buffer) + 4, output_content, output_byte_size);

        // Send the response.
        LOG_IF_ERROR(
            TRITONBACKEND_ResponseSend(
                responses[r], TRITONSERVER_RESPONSE_COMPLETE_FINAL,
                nullptr /* success */),
            "failed sending response"
        );

        // Report statistics for the successful request.
        uint64_t request_exec_end_ns = 0;
        SET_TIMESTAMP(request_exec_end_ns);
        LOG_IF_ERROR(
            TRITONBACKEND_ModelInstanceReportStatistics(
                instance_state->TritonModelInstance(), request, true /* success */,
                exec_start_ns, exec_start_ns, request_exec_end_ns, request_exec_end_ns),
            "failed reporting request statistics"
        );

        // Release each request as soon as we sent the corresponding response.
        LOG_IF_ERROR(
            TRITONBACKEND_RequestRelease(request, TRITONSERVER_REQUEST_RELEASE_ALL),
            "failed releasing request"
        );
    }

    // Report statistics for the entire batch of requests.
    uint64_t exec_end_ns = 0;
    SET_TIMESTAMP(exec_end_ns);
    LOG_IF_ERROR(
        TRITONBACKEND_ModelInstanceReportBatchStatistics(
            instance_state->TritonModelInstance(), total_batch_size,
            exec_start_ns, exec_start_ns, exec_end_ns, exec_end_ns),
        "failed reporting batch request statistics"
    );

    // Release Marian result.
    free_result(result);

    return nullptr;  // success
}

}  // extern "C"

}}} // namespace triton::backend::marian
