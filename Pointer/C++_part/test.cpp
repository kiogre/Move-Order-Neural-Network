#include <iostream>
#include <vector>
#include <onnxruntime_cxx_api.h>
#include <onnxruntime_c_api.h>

int main() {

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "JellyFishGPU");
    Ort::SessionOptions session_options;

    // Abilita l'Execution Provider CUDA (GPU 0)
    OrtCUDAProviderOptions cuda_options;
    cuda_options.device_id = 0;
    session_options.AppendExecutionProvider_CUDA(cuda_options);

    // Carica lo STESSO file .onnx generato prima
    Ort::Session session(env, "jellyfish_pointer.onnx", session_options);
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // Dimensioni dinamiche dell'input corrente
    int64_t batch_size = 1;
    int64_t n_mosse = 30; // Esempio: 30 mosse legali in questa posizione

    // 1. Dati di Input
    std::vector<float> board_data(batch_size * 13 * 8 * 8, 0.0f);
    std::vector<float> moves_data(batch_size * n_mosse * 46, 0.5f);
    std::vector<uint8_t> mask_data(batch_size * n_mosse, 1); // bool in ONNX = uint8_t (1 = true, 0 = false)

    // Shape dei tensori
    std::vector<int64_t> board_shape = {batch_size, 13, 8, 8};
    std::vector<int64_t> moves_shape = {batch_size, n_mosse, 46};
    std::vector<int64_t> mask_shape  = {batch_size, n_mosse};

    // 2. Creazione dei Tensori ONNX
    std::vector<Ort::Value> input_tensors;
    
    input_tensors.push_back(Ort::Value::CreateTensor<float>(
        memory_info, board_data.data(), board_data.size(), board_shape.data(), board_shape.size()));
        
    input_tensors.push_back(Ort::Value::CreateTensor<float>(
        memory_info, moves_data.data(), moves_data.size(), moves_shape.data(), moves_shape.size()));
        
    input_tensors.push_back(Ort::Value::CreateTensor<bool>(
        memory_info, reinterpret_cast<bool*>(mask_data.data()), mask_data.size(), mask_shape.data(), mask_shape.size()));

    // 3. Nomi dei nodi (devono corrispondere a quelli definiti in Python)
    const char* input_names[]  = {"board", "moves", "move_mask"};
    const char* output_names[] = {"logits", "probs", "value"};

    // 4. Invocazione dell'Inferenza
    auto output_tensors = session.Run(
        Ort::RunOptions{nullptr},
        input_names,
        input_tensors.data(),
        input_tensors.size(),
        output_names,
        3 // Numero di output attesi (logits, probs, value)
    );

    // 5. Estrazione dei risultati
    float* logits_ptr = output_tensors[0].GetTensorMutableData<float>();
    float* probs_ptr  = output_tensors[1].GetTensorMutableData<float>();
    float* value_ptr  = output_tensors[2].GetTensorMutableData<float>();

    std::cout << "Valore posizione stimato: " << value_ptr[0] << std::endl;
    std::cout << "Probabilità della prima mossa: " << probs_ptr[0] << std::endl;

    return 0;
}