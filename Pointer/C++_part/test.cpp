#include <iostream>
#include <vector>
#include <onnxruntime_cxx_api.h>
#include <onnxruntime_c_api.h>
#include "encoder.hpp"

int main() {
    // 1. Inserimento manuale della posizione FEN
    std::string fen_input;
    std::cout << "Inserisci FEN (Premi Invio per la posizione iniziale di default): ";
    std::getline(std::cin, fen_input);

    chess::Board board;
    if (fen_input.empty()) {
        board = chess::Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    } else {
        board = chess::Board(fen_input);
    }

    // 2. Generazione delle mosse legali reali con la libreria
    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board);

    int64_t n_mosse = moves.size();
    std::cout << "\nPosizione caricata correttamente." << std::endl;
    std::cout << "Mosse legali trovate: " << n_mosse << std::endl;

    if (n_mosse == 0) {
        std::cout << "Nessuna mossa legale disponibile (Scacco matto o Stallo)." << std::endl;
        return 0;
    }

    // 3. Allocazione ed encoding dei buffer
    int64_t batch_size = 1;
    std::vector<float> board_data(batch_size * 13 * 8 * 8);
    std::vector<float> moves_data(batch_size * n_mosse * 46);
    std::vector<uint8_t> mask_data(batch_size * n_mosse, 1); // Tutte le mosse generate sono legali (= 1)

    // Encoding della scacchiera
    encode_board_cpp(board, board_data.data());

    // Encoding delle mosse legali
    for (size_t i = 0; i < moves.size(); ++i) {
        encode_move_cpp(moves[i], board, moves_data.data() + (i * MOVE_VECTOR_DIM));
    }

    // 4. Inizializzazione Sessione ONNX Runtime
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "JellyFishGPU");
    Ort::SessionOptions session_options;

    OrtCUDAProviderOptions cuda_options;
    cuda_options.device_id = 0;
    session_options.AppendExecutionProvider_CUDA(cuda_options);

    Ort::Session session(env, "jellyfish_pointer.onnx", session_options);
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // 5. Creazione Tensori ONNX
    std::vector<int64_t> board_shape = {batch_size, 13, 8, 8};
    std::vector<int64_t> moves_shape = {batch_size, n_mosse, 46};
    std::vector<int64_t> mask_shape  = {batch_size, n_mosse};

    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(Ort::Value::CreateTensor<float>(
        memory_info, board_data.data(), board_data.size(), board_shape.data(), board_shape.size()));
    input_tensors.push_back(Ort::Value::CreateTensor<float>(
        memory_info, moves_data.data(), moves_data.size(), moves_shape.data(), moves_shape.size()));
    input_tensors.push_back(Ort::Value::CreateTensor<bool>(
        memory_info, reinterpret_cast<bool*>(mask_data.data()), mask_data.size(), mask_shape.data(), mask_shape.size()));

    const char* input_names[]  = {"board", "moves", "move_mask"};
    const char* output_names[] = {"logits", "probs", "value"};

    // 6. Esecuzione Inferenza GPU
    auto output_tensors = session.Run(
        Ort::RunOptions{nullptr},
        input_names,
        input_tensors.data(),
        input_tensors.size(),
        output_names,
        3
    );

    float* probs_ptr = output_tensors[1].GetTensorMutableData<float>();
    float* value_ptr = output_tensors[2].GetTensorMutableData<float>();

    // 7. Stampa dei Risultati per ciascuna mossa legale
    std::cout << "\n----------------------------------------" << std::endl;
    std::cout << "Valore valutazione posizione (Value): " << value_ptr[0] << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Probabilita' mosse (Policy output):" << std::endl;

    for (size_t i = 0; i < moves.size(); ++i) {
        std::string uci_move = chess::uci::moveToUci(moves[i]);
        std::cout << "  " << uci_move << " : " << (probs_ptr[i] * 100.0f) << "%" << std::endl;
    }

    return 0;
}