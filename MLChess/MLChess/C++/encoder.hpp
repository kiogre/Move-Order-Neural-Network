#include <vector>
#include <array>
#include "chess.hpp"

// Dizionario per la mappa dei pezzi espresso come array C++ ad accesso rapido O(1)
constexpr std::array<int, 128> make_piece_idx() {
    std::array<int, 128> idx = {};
    idx['p'] = idx['P'] = 0;
    idx['n'] = idx['N'] = 1;
    idx['b'] = idx['B'] = 2;
    idx['r'] = idx['R'] = 3;
    idx['q'] = idx['Q'] = 4;
    idx['k'] = idx['K'] = 5;
    return idx;
}
constexpr auto PIECE_IDX = make_piece_idx();

// Encoda direttamente l'oggetto chess::Board in un buffer float per ONNX (13 x 8 x 8)
void encode_board_cpp(const chess::Board& board, float* out_buffer) {
    // Azzera il buffer (13 * 64 = 832 floats)
    std::fill_n(out_buffer, 832, 0.0f);

    // Piano 12 (indice 12) impostato interamente a 1.0f
    
    std::fill_n(out_buffer + (12 * 64), 64, 1.0f);

    chess::Color stm = board.sideToMove();
    bool flip = (stm == chess::Color::BLACK);

    // Scansione di tutte le 64 case usando le API bitboard veloci della libreria
    for (int sq = 0; sq < 64; ++sq) {
        chess::Piece piece = board.at(chess::Square(sq));
        
        if (piece != chess::Piece::NONE) {
            char piece_char = static_cast<char>(piece); // Restituisce 'P', 'p', 'N', 'n', etc.
            
            int piece_idx = PIECE_IDX[static_cast<unsigned char>(piece_char)];
            bool is_white = (piece.color() == chess::Color::WHITE);
            bool is_current = (is_white != flip);

            int plane = is_current ? piece_idx : (piece_idx + 6);

            int rank = sq / 8;
            int file = sq % 8;

            int display_rank = flip ? (7 - rank) : rank;

            int tensor_idx = (plane * 64) + (display_rank * 8) + file;
            out_buffer[tensor_idx] = 1.0f;
        }
    }
}

constexpr size_t MOVE_VECTOR_DIM = 46;

constexpr std::array<int, 7> PIECE_TYPE_TO_IDX = {
    0, // PAWN
    1, // KNIGHT
    2, // BISHOP
    3, // ROOK
    4, // QUEEN
    5  // KING
};

// Codifica una singola mossa in un buffer float da 46 elementi
void encode_move_cpp(const chess::Move& move, const chess::Board& board, float* out_vec) {
    std::fill_n(out_vec, MOVE_VECTOR_DIM, 0.0f);

    bool flip = (board.sideToMove() == chess::Color::BLACK);

    int from_sq = move.from().index();
    int to_sq   = move.to().index();

    int from_row = from_sq / 8;
    int from_col = from_sq % 8;
    int to_row   = to_sq / 8;
    int to_col   = to_sq % 8;

    if (flip) {
        from_row = 7 - from_row;
        to_row   = 7 - to_row;
    }

    // Tipo di pezzo che muove
    chess::Piece piece = board.at(move.from());
    if (piece != chess::Piece::NONE) {
        int pt_idx = PIECE_TYPE_TO_IDX[static_cast<int>(piece.type())];
        out_vec[pt_idx] = 1.0f;
    }

    // Casa di partenza
    out_vec[6  + from_row] = 1.0f;
    out_vec[14 + from_col] = 1.0f;

    // Casa di arrivo
    out_vec[22 + to_row]   = 1.0f;
    out_vec[30 + to_col]   = 1.0f;

    // Cattura, En Passant, Arrocco
    out_vec[38] = board.isCapture(move) ? 1.0f : 0.0f;
    out_vec[39] = (move.typeOf() == chess::Move::ENPASSANT) ? 1.0f : 0.0f;
    out_vec[40] = (move.typeOf() == chess::Move::CASTLING)  ? 1.0f : 0.0f;

    // Promozione
    if (move.typeOf() == chess::Move::PROMOTION) {
        chess::PieceType promo = move.promotionType();
        int promo_idx = 0;
        if (promo == chess::PieceType::KNIGHT) promo_idx = 1;
        else if (promo == chess::PieceType::BISHOP) promo_idx = 2;
        else if (promo == chess::PieceType::ROOK)   promo_idx = 3;
        else if (promo == chess::PieceType::QUEEN)  promo_idx = 4;
        
        out_vec[41 + promo_idx] = 1.0f;
    } else {
        out_vec[41] = 1.0f; // promo_idx 0 (nessuna promozione)
    }
}

// Genera e encoda tutte le mosse legali direttamente in un vettore 1D piatto [N * 46]
std::vector<float> encode_legal_moves_cpp(const chess::Board& board, chess::Movelist& moves) {
    moves.clear();
    chess::movegen::legalmoves(moves, board);

    if (moves.empty()) {
        return {};
    }

    std::vector<float> encoded_buffer(moves.size() * MOVE_VECTOR_DIM);

    for (size_t i = 0; i < moves.size(); ++i) {
        encode_move_cpp(moves[i], board, encoded_buffer.data() + (i * MOVE_VECTOR_DIM));
    }

    return encoded_buffer;
}