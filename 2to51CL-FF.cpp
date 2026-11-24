#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <immintrin.h>
#include <fftw3.h>
#include <string>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <utility>
#include <cctype>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswresample/swresample.h>
#include <libavutil/avutil.h>
#include <libavutil/channel_layout.h>
#include <libavutil/opt.h>
#include <libavutil/audio_fifo.h>
#include <libavutil/mathematics.h>
#include <libavutil/dict.h>
}

using std::max;
using std::min;

// Check for modern FFmpeg AVChannelLayout API (FFmpeg 5.1+)
#if LIBAVUTIL_VERSION_INT >= AV_VERSION_INT(57, 28, 100)
#define USE_NEW_CH_LAYOUT 1
#else
#define USE_NEW_CH_LAYOUT 0
#endif

#ifdef _MSC_VER
#define NOINLINE_ATTR __declspec(noinline)
#else
#define NOINLINE_ATTR __attribute__((noinline))
#endif

constexpr size_t FFT_SIZE = 4096;
constexpr size_t HOP_SIZE = FFT_SIZE / 8; // 87.5% overlap
constexpr double PI = 3.14159265358979323846;
constexpr float RECOVERY_COEFF = 0.995f;

struct SurroundParams {
    double center_width = 0.25;       // 0.0 = exact center, 1.0 = entire stereo image
    double center_slope = 2.0;        // Falloff slope: 1.0 = gentle, 2.0 = standard, >= 4.0 = steep
    double surround_slope = 2.0;      // Falloff slope: 1.0 = gentle, 2.0 = standard, >= 4.0 = steep
    double surround_width = 0.33;      // 0.0 = extreme panned sounds only, 1.0 = entire L/R channels
    double surround_delay_ms = 0.0;  // Delay in milliseconds for ambient channels (typical: 15-30ms)
    double lfe_cutoff_hz = 120.0;     // LFE low-pass cutoff in Hz

    // Individual gain trim parameters in Decibels (dB) for all 6 output channels
    double gain_L_db = 0.0;
    double gain_R_db = 0.0;
    double gain_C_db = 0.0;
    double gain_lfe_db = 0.0;
    double gain_Ls_db = 0.0;
    double gain_Rs_db = 0.0;

    // Thresholds below which signal is attenuated (4:1 downward expander) and routed back to main L/R (dB)
    // Values <= -95.0 dB are considered OFF/disabled
    double center_thresh_db = -100.0;
    double surround_thresh_db = -100.0;

    // Audio Output Format & Settings
    std::string out_format;                   // Container format (empty = auto from output extension)
    std::string out_codec;                    // Codec name (empty = auto from container)
    std::string out_sample_fmt;               // Sample format name (empty = auto)
    std::string out_ch_layout = "auto";       // "auto", "5.1", "5.1(side)", "5.1(back)"
    int out_sample_rate = 0;                  // 0 = match input (or closest supported)
    int64_t out_bitrate = 0;                  // 0 = codec default (640k for lossy codecs)
    int out_bits_per_sample = 0;              // 0 = auto, or 16 / 24 / 32
    int out_compression_level = -1;           // -1 = codec default
    bool limiter_override = false;
    bool limiter_enabled = true;
    float limiter_threshold = 0.94f;
    std::vector<std::pair<std::string, std::string>> out_enc_options;
};

bool is_float_format(AVSampleFormat fmt) {
    return (fmt == AV_SAMPLE_FMT_FLT ||
        fmt == AV_SAMPLE_FMT_FLTP ||
        fmt == AV_SAMPLE_FMT_DBL ||
        fmt == AV_SAMPLE_FMT_DBLP);
}

static std::string to_lower_ascii(std::string s) {
    for (char& ch : s) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    return s;
}

static bool parse_bitrate(const std::string& str, int64_t& out_bps) {
    if (str.empty()) return false;
    try {
        size_t idx = 0;
        double val = std::stod(str, &idx);
        if (val <= 0.0) return false;
        std::string suffix = to_lower_ascii(str.substr(idx));
        if (suffix.empty()) {
            out_bps = static_cast<int64_t>(std::llround(val));
            return true;
        }
        if (suffix == "k" || suffix == "kbps") {
            out_bps = static_cast<int64_t>(std::llround(val * 1000.0));
            return true;
        }
        if (suffix == "m" || suffix == "mbps") {
            out_bps = static_cast<int64_t>(std::llround(val * 1000000.0));
            return true;
        }
    }
    catch (...) {
        return false;
    }
    return false;
}

void printUsage() {
    std::cout << "Usage: 2to51CL --input|-i <input_path> --output|-o <output_path> [options]\n"
        << "\nUpmix DSP Options:\n"
        << "  --cwidth|-cw <val>       Center spectral width fraction (0.0 to 1.0) [Default: 0.25]\n"
        << "  --cslope|-cs <val>       Center spectral slope fraction (0.5 to +4.0) [Default: 2.0]\n"
        << "  --lfeclp|-lp <hz>        LFE cutoff low-pass frequency in Hz [Default: 120.0]\n"
        << "  --swidth|-sw <val>       Surround width fraction (0.0 = extreme panned, 1.0 = entire L/R) [Default: 0.33]\n"
        << "  --sslope|-ss <val>       Surround spectral slope fraction (0.5 to +4.0) [Default: 2.0]\n"
        << "  --sdelay|-sd <ms>        Surround delay length in ms [Default: 0.0]\n"
        << "  --gainlr|-gf <dB>        Gain for left and right channels [Default: 0.0]\n"
        << "  --gainc|-gc <dB>         Gain for center channel [Default: 0.0]\n"
        << "  --gainlfe|-gl <dB>       Gain for LFE channel [Default: 0.0]\n"
        << "  --gainslsr|-gs <dB>      Gain for surround left and right channels [Default: 0.0]\n"
        << "  --cthresh|-ct <dB>       Threshold dB below which Center is attenuated (4:1) and routed to L/R [Default: -100.0 (off)]\n"
        << "  --sthresh|-st <dB>       Threshold dB below which Ls/Rs is attenuated (4:1) and routed to L/R [Default: -100.0 (off)]\n"
        << "\nAudio Output Format & Settings:(default same as input file)\n"
        << "  --format|-f <fmt>        Output container format (e.g. wav, flac, mp4, matroska, ac3, ogg) [Default: auto]\n"
        << "  --codec|-c <codec>       Output audio codec (e.g. pcm_s24le, pcm_f32le, flac, ac3, eac3, aac, libopus) [Default: auto]\n"
        << "  --samplerate|-sr <hz>    Output sample rate in Hz (e.g. 44100, 48000, 96000) [Default: match input]\n"
        << "  --samplefmt|-sf <fmt>    Output sample format (e.g. s16, s32, flt, fltp, dbl, dblp) [Default: auto]\n"
        << "  --bits|-b <16|24|32>     Bits per sample for PCM/FLAC/ALAC [Default: auto]\n"
        << "  --bitrate|-br <rate>     Output bitrate in bps, supports k/M suffix (e.g. 448k, 640k, 1.5M) [Default: 640k for lossy]\n"
        << "  --compression|-cl <lvl>  Encoder compression level (e.g. 0-12 for FLAC) [Default: codec default]\n"
        << "  --chlayout|-ch <layout>  5.1 channel layout: auto, 5.1, 5.1(side), 5.1(back) [Default: auto]\n"
        << "  --limiter|-lt <val|off>  Output limiter threshold (e.g. 0.94, 0.98, off) [Default: 0.94 int / off float]\n"
        << "  --encopt|-eo <k=v>       Codec-specific FFmpeg encoder option (repeatable, e.g. -eo dialnorm=-31)\n";
    std::cout << "\nGains and thresholds are in dB (e.g., 0.0 = unity, -6.0 = 0.5x, -100.0 = off).\n";
}

// ============================================================================
// 4:1 Soft-Knee Downward Expander Gain Calculation
// ============================================================================
inline double compute_expander_gain(double env, double thresh_db, double knee_high_lin) {
    constexpr double EXPANDER_SLOPE = 3.0; // 4:1 downward expander => ratio - 1 = 3.0
    constexpr double KNEE_DB = 6.0;        // +/- 3 dB soft-knee width around threshold
    constexpr double HALF_KNEE_DB = 3.0;
    constexpr double KNEE_COEFF = EXPANDER_SLOPE / (2.0 * KNEE_DB); // 0.25
    constexpr double LN_TO_DB = 8.6858896380650365530; // 20.0 / ln(10.0)
    constexpr double DB_TO_LN = 0.1151292546497022842; // ln(10.0) / 20.0
    constexpr double ENV_FLOOR = 1e-12; // -240 dB floor

    if (env >= knee_high_lin) {
        return 1.0;
    }
    if (env <= ENV_FLOOR) {
        return 0.0;
    }

    double env_db = std::log(env) * LN_TO_DB;
    double diff_db = env_db - thresh_db;

    if (diff_db >= HALF_KNEE_DB) {
        return 1.0;
    }

    double gain_db;
    if (diff_db <= -HALF_KNEE_DB) {
        gain_db = EXPANDER_SLOPE * diff_db;
    }
    else {
        double x = diff_db - HALF_KNEE_DB;
        gain_db = -KNEE_COEFF * (x * x);
    }

    return std::exp(gain_db * DB_TO_LN);
}

// ============================================================================
// Ring Buffer for Delay & Stage Latency Alignment
// ============================================================================
class DelayLine {
private:
    std::vector<double> buf;
    size_t write_idx = 0;
    size_t delay_len = 0;

public:
    void init(size_t delay_samples) {
        delay_len = delay_samples;
        buf.assign(delay_len + HOP_SIZE, 0.0);
        write_idx = 0;
    }

    void process(const double* in, double* out, size_t n) {
        if (delay_len == 0) {
            std::copy(in, in + n, out);
            return;
        }
        size_t cap = buf.size();
        for (size_t i = 0; i < n; ++i) {
            size_t read_idx = (write_idx + cap - delay_len) % cap;
            out[i] = buf[read_idx];
            buf[write_idx] = in[i];
            write_idx = (write_idx + 1) % cap;
        }
    }
};

// ============================================================================
// AVX Intrinsic Helpers (__m256d)
// ============================================================================
inline void apply_window_avx(const double* in, const double* win, double* out, size_t n) {
    for (size_t i = 0; i < n; i += 4) {
        __m256d v_in = _mm256_loadu_pd(&in[i]);
        __m256d v_win = _mm256_loadu_pd(&win[i]);
        _mm256_storeu_pd(&out[i], _mm256_mul_pd(v_in, v_win));
    }
}

inline void synthesis_ola_avx(const double* ifft_out, const double* win, double* accum, double scale, size_t n) {
    __m256d v_scale = _mm256_set1_pd(scale);
    for (size_t i = 0; i < n; i += 4) {
        __m256d v_out = _mm256_loadu_pd(&ifft_out[i]);
        __m256d v_win = _mm256_loadu_pd(&win[i]);
        __m256d v_acc = _mm256_loadu_pd(&accum[i]);
        __m256d v_prod = _mm256_mul_pd(v_out, v_win);
        _mm256_storeu_pd(&accum[i], _mm256_fmadd_pd(v_prod, v_scale, v_acc));
    }
}

inline void copy_with_gain_avx(const double* in, double* out, double gain, size_t n) {
    __m256d v_gain = _mm256_set1_pd(gain);
    for (size_t i = 0; i < n; i += 4) {
        __m256d v = _mm256_loadu_pd(&in[i]);
        _mm256_storeu_pd(&out[i], _mm256_mul_pd(v, v_gain));
    }
}

inline void subtract_avx(const double* ina, const double* inb, double* out, size_t n) {
    for (size_t i = 0; i < n; i += 4) {
        __m256d v1 = _mm256_loadu_pd(&ina[i]);
        __m256d v2 = _mm256_loadu_pd(&inb[i]);
        _mm256_storeu_pd(&out[i], _mm256_sub_pd(v1, v2));
    }
}

inline void load_deinterleave_4complex(const fftw_complex* ptr, __m256d& re, __m256d& im) {
    const double* p = reinterpret_cast<const double*>(ptr);
    __m256d v0 = _mm256_loadu_pd(p);
    __m256d v1 = _mm256_loadu_pd(p + 4);

    __m256d unpck_lo = _mm256_unpacklo_pd(v0, v1);
    __m256d unpck_hi = _mm256_unpackhi_pd(v0, v1);

    re = _mm256_permute4x64_pd(unpck_lo, _MM_SHUFFLE(3, 1, 2, 0));
    im = _mm256_permute4x64_pd(unpck_hi, _MM_SHUFFLE(3, 1, 2, 0));
}

inline void interleave_store_4complex(fftw_complex* ptr, __m256d re, __m256d im) {
    double* p = reinterpret_cast<double*>(ptr);

    __m256d unpck_lo = _mm256_unpacklo_pd(re, im);
    __m256d unpck_hi = _mm256_unpackhi_pd(re, im);

    __m256d v0 = _mm256_permute2f128_pd(unpck_lo, unpck_hi, 0x20);
    __m256d v1 = _mm256_permute2f128_pd(unpck_lo, unpck_hi, 0x31);

    _mm256_storeu_pd(p, v0);
    _mm256_storeu_pd(p + 4, v1);
}

// ============================================================================
// FFmpeg Audio Helper Functions
// ============================================================================

static const enum AVSampleFormat* get_supported_sample_fmts(const AVCodec* codec) {
    const enum AVSampleFormat* sample_fmts = nullptr;
#if LIBAVCODEC_VERSION_INT >= AV_VERSION_INT(61, 13, 100)
    avcodec_get_supported_config(nullptr, codec, AV_CODEC_CONFIG_SAMPLE_FORMAT, 0,
        reinterpret_cast<const void**>(&sample_fmts), nullptr);
#else
    sample_fmts = codec->sample_fmts;
#endif
    return sample_fmts;
}

static const int* get_supported_sample_rates(const AVCodec* codec) {
    const int* supported_samplerates = nullptr;
#if LIBAVCODEC_VERSION_INT >= AV_VERSION_INT(61, 13, 100)
    avcodec_get_supported_config(nullptr, codec, AV_CODEC_CONFIG_SAMPLE_RATE, 0,
        reinterpret_cast<const void**>(&supported_samplerates), nullptr);
#else
    supported_samplerates = codec->supported_samplerates;
#endif
    return supported_samplerates;
}

static enum AVSampleFormat select_sample_fmt(const AVCodec* codec, const std::string& requested_fmt, int requested_bits, bool& ok) {
    ok = true;
    const enum AVSampleFormat* sample_fmts = get_supported_sample_fmts(codec);

    if (!requested_fmt.empty()) {
        enum AVSampleFormat req = av_get_sample_fmt(requested_fmt.c_str());
        if (req == AV_SAMPLE_FMT_NONE) {
            std::cerr << "Error: Unknown sample format '" << requested_fmt << "'." << std::endl;
            ok = false;
            return AV_SAMPLE_FMT_NONE;
        }
        if (sample_fmts) {
            bool found = false;
            for (const enum AVSampleFormat* p = sample_fmts; *p != AV_SAMPLE_FMT_NONE; ++p) {
                if (*p == req) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                std::cerr << "Error: Sample format '" << requested_fmt << "' is not supported by encoder '" << codec->name << "'. Supported:";
                for (const enum AVSampleFormat* p = sample_fmts; *p != AV_SAMPLE_FMT_NONE; ++p) {
                    const char* name = av_get_sample_fmt_name(*p);
                    std::cerr << " " << (name ? name : "?");
                }
                std::cerr << std::endl;
                ok = false;
                return AV_SAMPLE_FMT_NONE;
            }
        }
        return req;
    }

    if (!sample_fmts) return AV_SAMPLE_FMT_FLTP;

    // If user explicitly requested 16-bit output (e.g. FLAC/ALAC), prefer 16-bit formats first
    if (requested_bits == 16) {
        const enum AVSampleFormat preferred_16[] = {
            AV_SAMPLE_FMT_S16,
            AV_SAMPLE_FMT_S16P,
            AV_SAMPLE_FMT_S32,
            AV_SAMPLE_FMT_S32P,
            AV_SAMPLE_FMT_FLTP,
            AV_SAMPLE_FMT_DBLP,
            AV_SAMPLE_FMT_FLT,
            AV_SAMPLE_FMT_DBL,
            AV_SAMPLE_FMT_NONE
        };
        for (int i = 0; preferred_16[i] != AV_SAMPLE_FMT_NONE; ++i) {
            for (const enum AVSampleFormat* p = sample_fmts; *p != AV_SAMPLE_FMT_NONE; ++p) {
                if (*p == preferred_16[i]) return *p;
            }
        }
    }

    const enum AVSampleFormat preferred[] = {
        AV_SAMPLE_FMT_FLTP,
        AV_SAMPLE_FMT_DBLP,
        AV_SAMPLE_FMT_S32,
        AV_SAMPLE_FMT_S32P,
        AV_SAMPLE_FMT_S16,
        AV_SAMPLE_FMT_S16P,
        AV_SAMPLE_FMT_FLT,
        AV_SAMPLE_FMT_DBL,
        AV_SAMPLE_FMT_NONE
    };

    for (int i = 0; preferred[i] != AV_SAMPLE_FMT_NONE; ++i) {
        for (const enum AVSampleFormat* p = sample_fmts; *p != AV_SAMPLE_FMT_NONE; ++p) {
            if (*p == preferred[i]) return *p;
        }
    }
    return sample_fmts[0];
}

static int select_sample_rate(const AVCodec* codec, int desired_rate, bool explicit_request) {
    const int* supported_samplerates = get_supported_sample_rates(codec);
    if (!supported_samplerates) return desired_rate;

    int best_rate = 0;
    for (const int* p = supported_samplerates; *p != 0; ++p) {
        if (*p == desired_rate) return desired_rate;
        if (best_rate == 0 || std::abs(*p - desired_rate) < std::abs(best_rate - desired_rate)) {
            best_rate = *p;
        }
    }
    if (explicit_request && best_rate != desired_rate) {
        std::cerr << "Warning: Requested sample rate " << desired_rate
            << " Hz is not supported by encoder '" << codec->name
            << "'. Using " << best_rate << " Hz instead." << std::endl;
    }
    return best_rate;
}

#if USE_NEW_CH_LAYOUT
static const AVChannelLayout* get_supported_ch_layouts(const AVCodec* codec) {
    const AVChannelLayout* ch_layouts = nullptr;
#if LIBAVCODEC_VERSION_INT >= AV_VERSION_INT(61, 13, 100)
    avcodec_get_supported_config(nullptr, codec, AV_CODEC_CONFIG_CHANNEL_LAYOUT, 0,
        reinterpret_cast<const void**>(&ch_layouts), nullptr);
#else
    ch_layouts = codec->ch_layouts;
#endif
    return ch_layouts;
}

static bool select_51_ch_layout(const AVCodec* codec, const std::string& requested, AVChannelLayout& out_layout) {
    const AVChannelLayout layout_side = AV_CHANNEL_LAYOUT_5POINT1;
    const AVChannelLayout layout_back = AV_CHANNEL_LAYOUT_5POINT1_BACK;
    const AVChannelLayout* supported = get_supported_ch_layouts(codec);

    auto is_supported = [&](const AVChannelLayout& target) -> bool {
        if (!supported) return true;
        for (const AVChannelLayout* p = supported; p->nb_channels != 0; ++p) {
            if (av_channel_layout_compare(p, &target) == 0) return true;
        }
        return false;
    };

    std::string req_lower = to_lower_ascii(requested);
    if (req_lower.empty() || req_lower == "auto") {
        if (is_supported(layout_side)) {
            av_channel_layout_copy(&out_layout, &layout_side);
            return true;
        }
        if (is_supported(layout_back)) {
            av_channel_layout_copy(&out_layout, &layout_back);
            return true;
        }
        std::cerr << "Error: Encoder '" << codec->name << "' does not support 5.1 surround layouts." << std::endl;
        return false;
    }

    const AVChannelLayout* chosen = nullptr;
    if (req_lower == "5.1" || req_lower == "5.1(side)" || req_lower == "side") {
        chosen = &layout_side;
    }
    else if (req_lower == "5.1(back)" || req_lower == "back") {
        chosen = &layout_back;
    }
    else {
        if (av_channel_layout_from_string(&out_layout, requested.c_str()) == 0 && out_layout.nb_channels == 6) {
            if (!is_supported(out_layout)) {
                std::cerr << "Warning: Channel layout '" << requested << "' may not be supported by encoder '" << codec->name << "'." << std::endl;
            }
            return true;
        }
        std::cerr << "Error: Invalid 5.1 channel layout '" << requested << "'. Use auto, 5.1(side), or 5.1(back)." << std::endl;
        return false;
    }

    if (!is_supported(*chosen)) {
        std::cerr << "Warning: Requested channel layout '" << requested << "' is not listed as supported by encoder '" << codec->name << "'." << std::endl;
    }
    av_channel_layout_copy(&out_layout, chosen);
    return true;
}
#else
static bool select_51_ch_layout_legacy(const AVCodec* codec, const std::string& requested, uint64_t& out_layout) {
    const uint64_t* supported = codec->channel_layouts;
    auto is_supported = [&](uint64_t target) -> bool {
        if (!supported) return true;
        for (const uint64_t* p = supported; *p != 0; ++p) {
            if (*p == target) return true;
        }
        return false;
    };

    std::string req_lower = to_lower_ascii(requested);
    if (req_lower.empty() || req_lower == "auto") {
        if (is_supported(AV_CH_LAYOUT_5POINT1)) {
            out_layout = AV_CH_LAYOUT_5POINT1;
            return true;
        }
        if (is_supported(AV_CH_LAYOUT_5POINT1_BACK)) {
            out_layout = AV_CH_LAYOUT_5POINT1_BACK;
            return true;
        }
        std::cerr << "Error: Encoder '" << codec->name << "' does not support 5.1 surround layouts." << std::endl;
        return false;
    }

    if (req_lower == "5.1" || req_lower == "5.1(side)" || req_lower == "side") {
        out_layout = AV_CH_LAYOUT_5POINT1;
    }
    else if (req_lower == "5.1(back)" || req_lower == "back") {
        out_layout = AV_CH_LAYOUT_5POINT1_BACK;
    }
    else {
        std::cerr << "Error: Invalid 5.1 channel layout '" << requested << "'. Use auto, 5.1(side), or 5.1(back)." << std::endl;
        return false;
    }

    if (!is_supported(out_layout)) {
        std::cerr << "Warning: Requested channel layout '" << requested << "' is not listed as supported by encoder '" << codec->name << "'." << std::endl;
    }
    return true;
}
#endif

static const AVCodec* find_output_encoder(const std::string& codec_name) {
    const AVCodec* enc = avcodec_find_encoder_by_name(codec_name.c_str());
    if (enc) return enc;

    const AVCodecDescriptor* desc = avcodec_descriptor_get_by_name(codec_name.c_str());
    if (desc) {
        enc = avcodec_find_encoder(desc->id);
        if (enc) return enc;
    }
    return nullptr;
}

static bool encode_and_write_frame(AVCodecContext* enc_ctx, AVFormatContext* fmt_ctx, AVStream* out_stream, AVFrame* frame) {
    int ret = avcodec_send_frame(enc_ctx, frame);
    if (ret < 0) {
        char err[AV_ERROR_MAX_STRING_SIZE] = { 0 };
        av_strerror(ret, err, sizeof(err));
        std::cerr << "Error sending frame to encoder: " << err << std::endl;
        return false;
    }

    AVPacket* pkt = av_packet_alloc();
    while (ret >= 0) {
        ret = avcodec_receive_packet(enc_ctx, pkt);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
            break;
        }
        else if (ret < 0) {
            av_packet_free(&pkt);
            return false;
        }

        av_packet_rescale_ts(pkt, enc_ctx->time_base, out_stream->time_base);
        pkt->stream_index = out_stream->index;

        ret = av_interleaved_write_frame(fmt_ctx, pkt);
        av_packet_unref(pkt);
        if (ret < 0) {
            av_packet_free(&pkt);
            return false;
        }
    }
    av_packet_free(&pkt);
    return true;
}

// ============================================================================
// Core DSP Up-mixer (Two-Stage Sequential Pipeline)
// ============================================================================

NOINLINE_ATTR bool process_audio(const char* in_file, const char* out_file, const SurroundParams& params) {
    // ------------------------------------------------------------------------
    // 1. Open Input (Demuxer & Decoder)
    // ------------------------------------------------------------------------
    AVFormatContext* fmt_ctx_in = nullptr;
    if (avformat_open_input(&fmt_ctx_in, in_file, nullptr, nullptr) < 0) {
        std::cerr << "Error: Could not open input file: " << in_file << std::endl;
        return false;
    }

    if (avformat_find_stream_info(fmt_ctx_in, nullptr) < 0) {
        std::cerr << "Error: Could not find stream information in input." << std::endl;
        avformat_close_input(&fmt_ctx_in);
        return false;
    }

    int audio_stream_idx = av_find_best_stream(fmt_ctx_in, AVMEDIA_TYPE_AUDIO, -1, -1, nullptr, 0);
    if (audio_stream_idx < 0) {
        std::cerr << "Error: No audio stream found in input." << std::endl;
        avformat_close_input(&fmt_ctx_in);
        return false;
    }

    AVStream* in_stream = fmt_ctx_in->streams[audio_stream_idx];
    const AVCodec* dec_codec = avcodec_find_decoder(in_stream->codecpar->codec_id);
    if (!dec_codec) {
        std::cerr << "Error: Unsupported input audio codec." << std::endl;
        avformat_close_input(&fmt_ctx_in);
        return false;
    }

    AVCodecContext* dec_ctx = avcodec_alloc_context3(dec_codec);
    avcodec_parameters_to_context(dec_ctx, in_stream->codecpar);
    if (avcodec_open2(dec_ctx, dec_codec, nullptr) < 0) {
        std::cerr << "Error: Could not open decoder." << std::endl;
        avcodec_free_context(&dec_ctx);
        avformat_close_input(&fmt_ctx_in);
        return false;
    }

#if USE_NEW_CH_LAYOUT
    if (dec_ctx->ch_layout.order == AV_CHANNEL_ORDER_UNSPEC) {
        av_channel_layout_default(&dec_ctx->ch_layout, dec_ctx->ch_layout.nb_channels);
    }
#endif

    const double sample_rate = static_cast<double>(dec_ctx->sample_rate);
    int64_t totalFrames = 0;
    if (in_stream->duration > 0) {
        totalFrames = av_rescale_q(in_stream->duration, in_stream->time_base, { 1, dec_ctx->sample_rate });
    }
    else if (fmt_ctx_in->duration > 0) {
        totalFrames = static_cast<int64_t>((fmt_ctx_in->duration * sample_rate) / AV_TIME_BASE);
    }

    // ------------------------------------------------------------------------
    // 2. Open Output (Muxer & Encoder)
    // ------------------------------------------------------------------------
    AVFormatContext* fmt_ctx_out = nullptr;
    const char* req_fmt_name = params.out_format.empty() ? nullptr : params.out_format.c_str();
    if (avformat_alloc_output_context2(&fmt_ctx_out, nullptr, req_fmt_name, out_file) < 0 || !fmt_ctx_out) {
        std::cerr << "Error: Could not deduce output format"
            << (req_fmt_name ? (std::string(" '") + req_fmt_name + "'") : (std::string(" from file extension: ") + out_file))
            << std::endl;
        avcodec_free_context(&dec_ctx);
        avformat_close_input(&fmt_ctx_in);
        return false;
    }

    const AVCodec* enc_codec = nullptr;
    if (!params.out_codec.empty()) {
        enc_codec = find_output_encoder(params.out_codec);
        if (!enc_codec) {
            std::cerr << "Error: Could not find audio encoder '" << params.out_codec << "'." << std::endl;
            avformat_free_context(fmt_ctx_out);
            avcodec_free_context(&dec_ctx);
            avformat_close_input(&fmt_ctx_in);
            return false;
        }
        if (enc_codec->type != AVMEDIA_TYPE_AUDIO) {
            std::cerr << "Error: Encoder '" << params.out_codec << "' is not an audio encoder." << std::endl;
            avformat_free_context(fmt_ctx_out);
            avcodec_free_context(&dec_ctx);
            avformat_close_input(&fmt_ctx_in);
            return false;
        }
    }
    else {
        enum AVCodecID out_codec_id = fmt_ctx_out->oformat->audio_codec;
        if (out_codec_id == AV_CODEC_ID_NONE) {
            out_codec_id = AV_CODEC_ID_PCM_S24LE;
        }
        if (params.out_bits_per_sample > 0 &&
            (out_codec_id == AV_CODEC_ID_PCM_S16LE || out_codec_id == AV_CODEC_ID_PCM_S24LE || out_codec_id == AV_CODEC_ID_PCM_S32LE)) {
            if (params.out_bits_per_sample == 16) out_codec_id = AV_CODEC_ID_PCM_S16LE;
            else if (params.out_bits_per_sample == 24) out_codec_id = AV_CODEC_ID_PCM_S24LE;
            else if (params.out_bits_per_sample == 32) out_codec_id = AV_CODEC_ID_PCM_S32LE;
        }
        else if (out_codec_id == AV_CODEC_ID_PCM_S16LE && avcodec_find_encoder(AV_CODEC_ID_PCM_S24LE)) {
            out_codec_id = AV_CODEC_ID_PCM_S24LE;
        }

        enc_codec = avcodec_find_encoder(out_codec_id);
        if (!enc_codec) {
            std::cerr << "Error: Could not find encoder for output format." << std::endl;
            avformat_free_context(fmt_ctx_out);
            avcodec_free_context(&dec_ctx);
            avformat_close_input(&fmt_ctx_in);
            return false;
        }
    }

    AVStream* out_stream = avformat_new_stream(fmt_ctx_out, nullptr);
    AVCodecContext* enc_ctx = avcodec_alloc_context3(enc_codec);

    const int desired_sr = (params.out_sample_rate > 0) ? params.out_sample_rate : dec_ctx->sample_rate;
    const int out_sample_rate = select_sample_rate(enc_codec, desired_sr, params.out_sample_rate > 0);
    enc_ctx->sample_rate = out_sample_rate;
    enc_ctx->time_base = { 1, out_sample_rate };

    bool fmt_ok = false;
    enc_ctx->sample_fmt = select_sample_fmt(enc_codec, params.out_sample_fmt, params.out_bits_per_sample, fmt_ok);
    if (!fmt_ok) {
        avcodec_free_context(&enc_ctx);
        avformat_free_context(fmt_ctx_out);
        avcodec_free_context(&dec_ctx);
        avformat_close_input(&fmt_ctx_in);
        return false;
    }

    if (params.out_bits_per_sample > 0) {
        enc_ctx->bits_per_raw_sample = params.out_bits_per_sample;
    }
    if (params.out_compression_level >= 0) {
        enc_ctx->compression_level = params.out_compression_level;
    }

#if USE_NEW_CH_LAYOUT
    if (!select_51_ch_layout(enc_codec, params.out_ch_layout, enc_ctx->ch_layout)) {
        avcodec_free_context(&enc_ctx);
        avformat_free_context(fmt_ctx_out);
        avcodec_free_context(&dec_ctx);
        avformat_close_input(&fmt_ctx_in);
        return false;
    }
#else
    enc_ctx->channels = 6;
    if (!select_51_ch_layout_legacy(enc_codec, params.out_ch_layout, enc_ctx->channel_layout)) {
        avcodec_free_context(&enc_ctx);
        avformat_free_context(fmt_ctx_out);
        avcodec_free_context(&dec_ctx);
        avformat_close_input(&fmt_ctx_in);
        return false;
    }
#endif

    if (params.out_bitrate > 0) {
        enc_ctx->bit_rate = params.out_bitrate;
    }
    else if (enc_codec->id == AV_CODEC_ID_AAC || enc_codec->id == AV_CODEC_ID_AC3 ||
        enc_codec->id == AV_CODEC_ID_EAC3 || enc_codec->id == AV_CODEC_ID_OPUS ||
        enc_codec->id == AV_CODEC_ID_VORBIS || enc_codec->id == AV_CODEC_ID_DTS) {
        enc_ctx->bit_rate = 640000;
    }

    if (fmt_ctx_out->oformat->flags & AVFMT_GLOBALHEADER) {
        enc_ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    }

    AVDictionary* enc_opts = nullptr;
    for (const auto& kv : params.out_enc_options) {
        av_dict_set(&enc_opts, kv.first.c_str(), kv.second.c_str(), 0);
    }

    if (avcodec_open2(enc_ctx, enc_codec, &enc_opts) < 0) {
        std::cerr << "Error: Could not open encoder '" << enc_codec->name << "'. Note: Target codec must support 5.1 surround and the chosen settings." << std::endl;
        av_dict_free(&enc_opts);
        avcodec_free_context(&enc_ctx);
        avformat_free_context(fmt_ctx_out);
        avcodec_free_context(&dec_ctx);
        avformat_close_input(&fmt_ctx_in);
        return false;
    }

    if (enc_opts) {
        const AVDictionaryEntry* entry = nullptr;
        while ((entry = av_dict_get(enc_opts, "", entry, AV_DICT_IGNORE_SUFFIX))) {
            std::cerr << "Warning: Unrecognized encoder option '" << entry->key << "=" << entry->value << "' for codec '" << enc_codec->name << "'." << std::endl;
        }
        av_dict_free(&enc_opts);
    }

    avcodec_parameters_from_context(out_stream->codecpar, enc_ctx);
    out_stream->time_base = { 1, enc_ctx->sample_rate };

    if (!(fmt_ctx_out->oformat->flags & AVFMT_NOFILE)) {
        if (avio_open(&fmt_ctx_out->pb, out_file, AVIO_FLAG_WRITE) < 0) {
            std::cerr << "Error: Could not open output file for writing: " << out_file << std::endl;
            avcodec_free_context(&enc_ctx);
            avformat_free_context(fmt_ctx_out);
            avcodec_free_context(&dec_ctx);
            avformat_close_input(&fmt_ctx_in);
            return false;
        }
    }

    if (avformat_write_header(fmt_ctx_out, nullptr) < 0) {
        std::cerr << "Error: Could not write output header (container may not support the selected codec)." << std::endl;
        if (!(fmt_ctx_out->oformat->flags & AVFMT_NOFILE)) {
            avio_closep(&fmt_ctx_out->pb);
        }
        avcodec_free_context(&enc_ctx);
        avformat_free_context(fmt_ctx_out);
        avcodec_free_context(&dec_ctx);
        avformat_close_input(&fmt_ctx_in);
        return false;
    }

    // Determine effective output limiter threshold based on output sample format unless overridden
    const bool use_limiter = params.limiter_override
        ? params.limiter_enabled
        : !is_float_format(enc_ctx->sample_fmt);
    const float limiter_threshold = params.limiter_override
        ? params.limiter_threshold
        : 0.94f;

    const char* out_sfmt_name = av_get_sample_fmt_name(enc_ctx->sample_fmt);
    std::cout << " - Output Container   : " << fmt_ctx_out->oformat->name << std::endl;
    std::cout << " - Output Encoder     : " << enc_codec->name << std::endl;
    std::cout << " - Output Sample Rate : " << enc_ctx->sample_rate << " Hz" << std::endl;
    std::cout << " - Output Sample Fmt  : " << (out_sfmt_name ? out_sfmt_name : "unknown") << std::endl;
    if (enc_ctx->bit_rate > 0) {
        std::cout << " - Output Bitrate     : " << (enc_ctx->bit_rate / 1000) << " kbps" << std::endl;
    }
    std::cout << " - Output Limiter     : "
        << (use_limiter ? (std::to_string(limiter_threshold)) : "off") << std::endl;

    // ------------------------------------------------------------------------
    // 3. Audio Resampling Contexts & Audio FIFOs
    // ------------------------------------------------------------------------
    SwrContext* swr_in = nullptr;
#if USE_NEW_CH_LAYOUT
    AVChannelLayout stereo_layout = AV_CHANNEL_LAYOUT_STEREO;
    swr_alloc_set_opts2(&swr_in,
        &stereo_layout, AV_SAMPLE_FMT_DBL, dec_ctx->sample_rate,
        &dec_ctx->ch_layout, dec_ctx->sample_fmt, dec_ctx->sample_rate,
        0, nullptr);
#else
    swr_in = swr_alloc_set_opts(nullptr,
        AV_CH_LAYOUT_STEREO, AV_SAMPLE_FMT_DBL, dec_ctx->sample_rate,
        dec_ctx->channel_layout ? dec_ctx->channel_layout : av_get_default_channel_layout(dec_ctx->channels),
        dec_ctx->sample_fmt, dec_ctx->sample_rate,
        0, nullptr);
#endif
    if (!swr_in || swr_init(swr_in) < 0) {
        std::cerr << "Error: Could not initialize input resampler." << std::endl;
        swr_free(&swr_in);
        if (!(fmt_ctx_out->oformat->flags & AVFMT_NOFILE)) avio_closep(&fmt_ctx_out->pb);
        avcodec_free_context(&enc_ctx);
        avformat_free_context(fmt_ctx_out);
        avcodec_free_context(&dec_ctx);
        avformat_close_input(&fmt_ctx_in);
        return false;
    }

    SwrContext* swr_out = nullptr;
#if USE_NEW_CH_LAYOUT
    AVChannelLayout surr_layout = AV_CHANNEL_LAYOUT_5POINT1;
    swr_alloc_set_opts2(&swr_out,
        &enc_ctx->ch_layout, enc_ctx->sample_fmt, enc_ctx->sample_rate,
        &surr_layout, AV_SAMPLE_FMT_DBL, static_cast<int>(sample_rate),
        0, nullptr);
#else
    swr_out = swr_alloc_set_opts(nullptr,
        enc_ctx->channel_layout ? enc_ctx->channel_layout : av_get_default_channel_layout(enc_ctx->channels),
        enc_ctx->sample_fmt, enc_ctx->sample_rate,
        AV_CH_LAYOUT_5POINT1, AV_SAMPLE_FMT_DBL, static_cast<int>(sample_rate),
        0, nullptr);
#endif
    if (!swr_out || swr_init(swr_out) < 0) {
        std::cerr << "Error: Could not initialize output resampler." << std::endl;
        swr_free(&swr_out);
        swr_free(&swr_in);
        if (!(fmt_ctx_out->oformat->flags & AVFMT_NOFILE)) avio_closep(&fmt_ctx_out->pb);
        avcodec_free_context(&enc_ctx);
        avformat_free_context(fmt_ctx_out);
        avcodec_free_context(&dec_ctx);
        avformat_close_input(&fmt_ctx_in);
        return false;
    }

    AVAudioFifo* in_fifo = av_audio_fifo_alloc(AV_SAMPLE_FMT_DBL, 2, static_cast<int>(HOP_SIZE * 8));
    AVAudioFifo* out_fifo = av_audio_fifo_alloc(enc_ctx->sample_fmt, 6, static_cast<int>(HOP_SIZE * 8));

    // ------------------------------------------------------------------------
    // 4. DSP Parameters and Buffers Setup
    // ------------------------------------------------------------------------
    const double g_lin_L = std::pow(10.0, params.gain_L_db / 20.0);
    const double g_lin_R = std::pow(10.0, params.gain_R_db / 20.0);
    const double g_lin_C = std::pow(10.0, params.gain_C_db / 20.0);
    const double g_lin_LFE = std::pow(10.0, params.gain_lfe_db / 20.0);
    const double g_lin_Ls = std::pow(10.0, params.gain_Ls_db / 20.0);
    const double g_lin_Rs = std::pow(10.0, params.gain_Rs_db / 20.0);

    // Dynamic thresholds (any setting <= -95.0 dB is treated as disabled)
    const bool c_thresh_active = (params.center_thresh_db > -95.0);
    const bool s_thresh_active = (params.surround_thresh_db > -95.0);

    // Upper soft-knee bounds (+3 dB above threshold; signals above this get 1.0 unity gain)
    const double c_knee_high = std::pow(10.0, (params.center_thresh_db + 3.0) / 20.0);
    const double s_knee_high = std::pow(10.0, (params.surround_thresh_db + 3.0) / 20.0);

    // Time constants for envelope tracking and gain smoothing
    constexpr double C_ATTACK_SEC = 0.003;        // 3 ms
    constexpr double C_RELEASE_SEC = 0.150;       // 150 ms
    constexpr double C_GAIN_SMOOTH_SEC = 0.005;   // 5 ms

    constexpr double S_ATTACK_SEC = 0.015;        // 15 ms
    constexpr double S_RELEASE_SEC = 0.300;       // 300 ms
    constexpr double S_GAIN_SMOOTH_SEC = 0.010;   // 10 ms

    const double alpha_c_attack = std::exp(-1.0 / (C_ATTACK_SEC * sample_rate));
    const double alpha_c_release = std::exp(-1.0 / (C_RELEASE_SEC * sample_rate));
    const double alpha_c_gain = std::exp(-1.0 / (C_GAIN_SMOOTH_SEC * sample_rate));

    const double alpha_s_attack = std::exp(-1.0 / (S_ATTACK_SEC * sample_rate));
    const double alpha_s_release = std::exp(-1.0 / (S_RELEASE_SEC * sample_rate));
    const double alpha_s_gain = std::exp(-1.0 / (S_GAIN_SMOOTH_SEC * sample_rate));

    // Continuous envelope and expander gain state (preserved across HOP_SIZE blocks)
    double env_C = 0.0;
    double exp_gain_C = 1.0;
    double env_Ls = 0.0;
    double exp_gain_Ls = 1.0;
    double env_Rs = 0.0;
    double exp_gain_Rs = 1.0;

    // Surround delay lines
    const size_t surround_delay_samples = static_cast<size_t>(
        max(0.0, std::round(params.surround_delay_ms * sample_rate / 1000.0))
        );
    DelayLine delay_line_Ls, delay_line_Rs;
    delay_line_Ls.init(surround_delay_samples);
    delay_line_Rs.init(surround_delay_samples);

    // Stage 2 latency synchronization delays for Stage 1 LFE and gated Center
    constexpr size_t STAGE_LATENCY_SAMPLES = FFT_SIZE - HOP_SIZE;
    DelayLine sync_delay_LFE, sync_delay_C;
    sync_delay_LFE.init(STAGE_LATENCY_SAMPLES);
    sync_delay_C.init(STAGE_LATENCY_SAMPLES);

    std::vector<double> window(FFT_SIZE);
    for (size_t i = 0; i < FFT_SIZE; ++i) {
        window[i] = std::sin((i + 0.5) * PI / FFT_SIZE);
    }

    const size_t spec_size = FFT_SIZE / 2 + 1;

    // LFE low-pass and matching complementary high-pass weights at the same cutoff frequency
    std::vector<double> lfe_weights(spec_size);
    std::vector<double> hpf_weights(spec_size);
    for (size_t k = 0; k < spec_size; ++k) {
        double freq = static_cast<double>(k) * sample_rate / FFT_SIZE;
        double f_ratio = freq / max(10.0, params.lfe_cutoff_hz);
        lfe_weights[k] = 1.0 / std::sqrt(1.0 + std::pow(f_ratio, 8.0));
        hpf_weights[k] = 1.0 - lfe_weights[k];
    }

    // FFTW Allocations - Stage 1 (LFE, TempL, TempR, Center)
    double* in_L = fftw_alloc_real(FFT_SIZE);
    double* in_R = fftw_alloc_real(FFT_SIZE);
    fftw_complex* spec_L = fftw_alloc_complex(spec_size);
    fftw_complex* spec_R = fftw_alloc_complex(spec_size);

    fftw_complex* spec_LFE = fftw_alloc_complex(spec_size);
    fftw_complex* spec_TempL = fftw_alloc_complex(spec_size);
    fftw_complex* spec_TempR = fftw_alloc_complex(spec_size);
    fftw_complex* spec_C = fftw_alloc_complex(spec_size);

    double* out_LFE = fftw_alloc_real(FFT_SIZE);
    double* out_TempL = fftw_alloc_real(FFT_SIZE);
    double* out_TempR = fftw_alloc_real(FFT_SIZE);
    double* out_C = fftw_alloc_real(FFT_SIZE);

    fftw_plan p_fwd_L = fftw_plan_dft_r2c_1d((int)FFT_SIZE, in_L, spec_L, FFTW_ESTIMATE);
    fftw_plan p_fwd_R = fftw_plan_dft_r2c_1d((int)FFT_SIZE, in_R, spec_R, FFTW_ESTIMATE);

    fftw_plan p_inv_LFE = fftw_plan_dft_c2r_1d((int)FFT_SIZE, spec_LFE, out_LFE, FFTW_ESTIMATE);
    fftw_plan p_inv_TempL = fftw_plan_dft_c2r_1d((int)FFT_SIZE, spec_TempL, out_TempL, FFTW_ESTIMATE);
    fftw_plan p_inv_TempR = fftw_plan_dft_c2r_1d((int)FFT_SIZE, spec_TempR, out_TempR, FFTW_ESTIMATE);
    fftw_plan p_inv_C = fftw_plan_dft_c2r_1d((int)FFT_SIZE, spec_C, out_C, FFTW_ESTIMATE);

    // FFTW Allocations - Stage 2 (Post-Center TempL/TempR -> Ambient L/R)
    double* in2_L = fftw_alloc_real(FFT_SIZE);
    double* in2_R = fftw_alloc_real(FFT_SIZE);
    fftw_complex* spec2_L = fftw_alloc_complex(spec_size);
    fftw_complex* spec2_R = fftw_alloc_complex(spec_size);

    fftw_complex* spec_SurroundL = fftw_alloc_complex(spec_size);
    fftw_complex* spec_SurroundR = fftw_alloc_complex(spec_size);

    double* out_SurroundL = fftw_alloc_real(FFT_SIZE);
    double* out_SurroundR = fftw_alloc_real(FFT_SIZE);

    fftw_plan p2_fwd_L = fftw_plan_dft_r2c_1d((int)FFT_SIZE, in2_L, spec2_L, FFTW_ESTIMATE);
    fftw_plan p2_fwd_R = fftw_plan_dft_r2c_1d((int)FFT_SIZE, in2_R, spec2_R, FFTW_ESTIMATE);

    fftw_plan p_inv_SurroundL = fftw_plan_dft_c2r_1d((int)FFT_SIZE, spec_SurroundL, out_SurroundL, FFTW_ESTIMATE);
    fftw_plan p_inv_SurroundR = fftw_plan_dft_c2r_1d((int)FFT_SIZE, spec_SurroundR, out_SurroundR, FFTW_ESTIMATE);

    // OLA accumulators and input delay lines
    std::vector<double> ola_LFE(FFT_SIZE, 0.0);
    std::vector<double> ola_TempL(FFT_SIZE, 0.0);
    std::vector<double> ola_TempR(FFT_SIZE, 0.0);
    std::vector<double> ola_C(FFT_SIZE, 0.0);
    std::vector<double> ola_SurroundL(FFT_SIZE, 0.0);
    std::vector<double> ola_SurroundR(FFT_SIZE, 0.0);

    std::vector<double> delay_L(FFT_SIZE, 0.0);
    std::vector<double> delay_R(FFT_SIZE, 0.0);
    std::vector<double> delay2_L(FFT_SIZE, 0.0);
    std::vector<double> delay2_R(FFT_SIZE, 0.0);

    // Per-hop working buffers
    std::vector<double> gated_C(HOP_SIZE);
    std::vector<double> temp_L_postC(HOP_SIZE);
    std::vector<double> temp_R_postC(HOP_SIZE);

    std::vector<double> aligned_LFE(HOP_SIZE);
    std::vector<double> aligned_C(HOP_SIZE);

    std::vector<double> gated_SurroundL(HOP_SIZE);
    std::vector<double> gated_SurroundR(HOP_SIZE);
    std::vector<double> delayed_SurroundL(HOP_SIZE);
    std::vector<double> delayed_SurroundR(HOP_SIZE);

    std::vector<double> final_L_sub(HOP_SIZE);
    std::vector<double> final_R_sub(HOP_SIZE);
    std::vector<double> final_C(HOP_SIZE);
    std::vector<double> final_LFE(HOP_SIZE);
    std::vector<double> final_Ls(HOP_SIZE);
    std::vector<double> final_Rs(HOP_SIZE);

    std::vector<double> interleaved_in(HOP_SIZE * 2);
    std::vector<double> interleaved_out(HOP_SIZE * 6);

    // Center & Surround mask smoothing buffers
    std::vector<double> raw_center_mask(spec_size, 0.0);
    std::vector<double> prev_center_mask(spec_size, 0.0);

    std::vector<double> raw_surround_mask(spec_size, 0.0);
    std::vector<double> spec_smoothed_mask(spec_size, 0.0);
    std::vector<double> prev_surround_mask(spec_size, 0.0);

    const double overlap_factor = static_cast<double>(FFT_SIZE) / static_cast<double>(HOP_SIZE);
    const double ifft_scale = 1.0 / (static_cast<double>(FFT_SIZE) * (overlap_factor * 0.5));
    const double eps = 1e-12;
    const double effective_width = max(0.005, params.center_width);
    const double cslope_power = 2.0 * max(0.1, params.center_slope);
    const double sslope_power = 2.0 * max(0.1, params.surround_slope);
    const double effective_surround_width = max(0.005, params.surround_width);
    const double temporal_alpha = std::pow(0.837, static_cast<double>(HOP_SIZE) / 1024.0);

    const __m256d v_half = _mm256_set1_pd(0.5);
    const __m256d v_one = _mm256_set1_pd(1.0);
    const __m256d v_zero = _mm256_setzero_pd();
    const __m256d v_eps = _mm256_set1_pd(eps);
    const __m256d v_inv_width = _mm256_set1_pd(1.0 / effective_width);
    const __m256d v_abs_mask = _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FFFFFFFFFFFFFFF));

    const size_t vec_bins = spec_size - (spec_size % 4);

    float currentLimiterGain = 1.0f;
    int64_t framesProcessed = 0;
    int warmup_blocks_1 = static_cast<int>(FFT_SIZE / HOP_SIZE) - 1;
    int warmup_blocks_2 = static_cast<int>(FFT_SIZE / HOP_SIZE) - 1;
    size_t flush_blocks_remaining = static_cast<size_t>(warmup_blocks_1 + warmup_blocks_2);

    auto last_update = std::chrono::steady_clock::now();
    float CValLast = 0.0f, LRValLast = 0.0f, LFEValLast = 0.0f, LRsValLast = 0.0f, MaxValLast = 0.0f;

    const int enc_frame_size = enc_ctx->frame_size > 0 ? enc_ctx->frame_size : static_cast<int>(HOP_SIZE);
    AVFrame* dec_frame = av_frame_alloc();
    AVPacket* in_pkt = av_packet_alloc();
    int64_t pts_counter = 0;

    double* dbl_tmp = nullptr;
    int dbl_tmp_capacity = 0;

    uint8_t** converted_resamp_buf = nullptr;
    int converted_resamp_linesize = 0;
    int max_resampled_samples = static_cast<int>(av_rescale_rnd(HOP_SIZE, enc_ctx->sample_rate, static_cast<int>(sample_rate), AV_ROUND_UP)) + 256;
    av_samples_alloc_array_and_samples(&converted_resamp_buf, &converted_resamp_linesize, 6, max_resampled_samples, enc_ctx->sample_fmt, 0);

    bool input_eof = false;

    auto shift_ola = [](std::vector<double>& buf) {
        std::copy(buf.begin() + HOP_SIZE, buf.end(), buf.begin());
        std::fill(buf.end() - HOP_SIZE, buf.end(), 0.0);
    };

    // ------------------------------------------------------------------------
    // 5. Main Processing Loop
    // ------------------------------------------------------------------------
    while (true) {
        // Feed in_fifo from demuxer/decoder until it has >= HOP_SIZE samples or EOF
        while (av_audio_fifo_size(in_fifo) < static_cast<int>(HOP_SIZE) && !input_eof) {
            int ret = av_read_frame(fmt_ctx_in, in_pkt);
            if (ret < 0) {
                avcodec_send_packet(dec_ctx, nullptr);
                input_eof = true;
            }
            else if (in_pkt->stream_index == audio_stream_idx) {
                avcodec_send_packet(dec_ctx, in_pkt);
                av_packet_unref(in_pkt);
            }
            else {
                av_packet_unref(in_pkt);
                continue;
            }

            while (true) {
                ret = avcodec_receive_frame(dec_ctx, dec_frame);
                if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) break;
                if (ret < 0) break;

                if (dec_frame->nb_samples > dbl_tmp_capacity) {
                    av_freep(&dbl_tmp);
                    dbl_tmp_capacity = dec_frame->nb_samples;
                    av_samples_alloc(reinterpret_cast<uint8_t**>(&dbl_tmp), nullptr, 2, dbl_tmp_capacity, AV_SAMPLE_FMT_DBL, 0);
                }

                int converted = swr_convert(swr_in, reinterpret_cast<uint8_t**>(&dbl_tmp), dbl_tmp_capacity,
                    const_cast<const uint8_t**>(dec_frame->extended_data), dec_frame->nb_samples);
                if (converted > 0) {
                    av_audio_fifo_write(in_fifo, reinterpret_cast<void**>(&dbl_tmp), converted);
                }
                av_frame_unref(dec_frame);
            }
        }

        int available = av_audio_fifo_size(in_fifo);
        size_t samples_to_read = 0;
        if (available > 0) {
            samples_to_read = min<size_t>(HOP_SIZE, available);
            void* in_ptrs[1] = { interleaved_in.data() };
            av_audio_fifo_read(in_fifo, in_ptrs, static_cast<int>(samples_to_read));

            if (samples_to_read < HOP_SIZE) {
                std::fill(interleaved_in.begin() + samples_to_read * 2, interleaved_in.end(), 0.0);
            }
        }
        else if (input_eof) {
            if (flush_blocks_remaining == 0) {
                break;
            }
            std::fill(interleaved_in.begin(), interleaved_in.end(), 0.0);
            --flush_blocks_remaining;
        }

        // ====================================================================
        // STAGE 1: LFE + High-Passed Temporary L/R + Center Extraction
        // ====================================================================
        std::copy(delay_L.begin() + HOP_SIZE, delay_L.end(), delay_L.begin());
        std::copy(delay_R.begin() + HOP_SIZE, delay_R.end(), delay_R.begin());

        for (size_t i = 0; i < HOP_SIZE; ++i) {
            delay_L[(FFT_SIZE - HOP_SIZE) + i] = interleaved_in[i * 2 + 0];
            delay_R[(FFT_SIZE - HOP_SIZE) + i] = interleaved_in[i * 2 + 1];
        }

        apply_window_avx(delay_L.data(), window.data(), in_L, FFT_SIZE);
        apply_window_avx(delay_R.data(), window.data(), in_R, FFT_SIZE);

        fftw_execute(p_fwd_L);
        fftw_execute(p_fwd_R);

        // Stage 1 Pass 1: LFE from original stereo, HPF Temporary L/R, and Raw Center Mask
        for (size_t k = 0; k < vec_bins; k += 4) {
            __m256d v_L_re, v_L_im, v_R_re, v_R_im;
            load_deinterleave_4complex(&spec_L[k], v_L_re, v_L_im);
            load_deinterleave_4complex(&spec_R[k], v_R_re, v_R_im);

            __m256d v_lfe_w = _mm256_loadu_pd(&lfe_weights[k]);
            __m256d v_hpf_w = _mm256_loadu_pd(&hpf_weights[k]);

            // LFE from original stereo
            __m256d v_mono_re = _mm256_mul_pd(_mm256_add_pd(v_L_re, v_R_re), v_half);
            __m256d v_mono_im = _mm256_mul_pd(_mm256_add_pd(v_L_im, v_R_im), v_half);
            __m256d v_LFE_re = _mm256_mul_pd(v_mono_re, v_lfe_w);
            __m256d v_LFE_im = _mm256_mul_pd(v_mono_im, v_lfe_w);
            interleave_store_4complex(&spec_LFE[k], v_LFE_re, v_LFE_im);

            // High-pass filter on original stereo -> Temporary L and R
            __m256d v_TempL_re = _mm256_mul_pd(v_L_re, v_hpf_w);
            __m256d v_TempL_im = _mm256_mul_pd(v_L_im, v_hpf_w);
            __m256d v_TempR_re = _mm256_mul_pd(v_R_re, v_hpf_w);
            __m256d v_TempR_im = _mm256_mul_pd(v_R_im, v_hpf_w);
            interleave_store_4complex(&spec_TempL[k], v_TempL_re, v_TempL_im);
            interleave_store_4complex(&spec_TempR[k], v_TempR_re, v_TempR_im);

            // Raw Center Mask from Temporary L and R
            if (params.center_width >= 0.999) {
                raw_center_mask[k + 0] = 1.0;
                raw_center_mask[k + 1] = 1.0;
                raw_center_mask[k + 2] = 1.0;
                raw_center_mask[k + 3] = 1.0;
            }
            else {
                __m256d v_P_L = _mm256_fmadd_pd(v_TempL_re, v_TempL_re, _mm256_mul_pd(v_TempL_im, v_TempL_im));
                __m256d v_P_R = _mm256_fmadd_pd(v_TempR_re, v_TempR_re, _mm256_mul_pd(v_TempR_im, v_TempR_im));
                __m256d v_mag_L = _mm256_sqrt_pd(v_P_L);
                __m256d v_mag_R = _mm256_sqrt_pd(v_P_R);

                __m256d v_diff = _mm256_and_pd(_mm256_sub_pd(v_mag_L, v_mag_R), v_abs_mask);
                __m256d v_sum = _mm256_add_pd(_mm256_add_pd(v_mag_L, v_mag_R), v_eps);
                __m256d v_pan = _mm256_div_pd(v_diff, v_sum);
                __m256d v_norm_pan = _mm256_mul_pd(v_pan, v_inv_width);

                // Positive phase correlation (ICC) to reject uncorrelated / out-of-phase stereo
                __m256d v_dot = _mm256_fmadd_pd(v_TempL_re, v_TempR_re, _mm256_mul_pd(v_TempL_im, v_TempR_im));
                __m256d v_tot_pwr = _mm256_add_pd(_mm256_add_pd(v_P_L, v_P_R), v_eps);
                __m256d v_icc = _mm256_max_pd(v_zero, _mm256_div_pd(_mm256_add_pd(v_dot, v_dot), v_tot_pwr));

                alignas(32) double npan[4];
                alignas(32) double r_icc[4];
                _mm256_storeu_pd(npan, v_norm_pan);
                _mm256_storeu_pd(r_icc, v_icc);

                for (int b = 0; b < 4; ++b) {
                    double cmask = 1.0 / (1.0 + std::pow(npan[b], cslope_power));
                    raw_center_mask[k + b] = cmask * r_icc[b];
                }
            }
        }
        // Stage 1 Pass 1 Scalar Tail
        for (size_t k = vec_bins; k < spec_size; ++k) {
            double L_re = spec_L[k][0], L_im = spec_L[k][1];
            double R_re = spec_R[k][0], R_im = spec_R[k][1];

            double mono_re = 0.5 * (L_re + R_re);
            double mono_im = 0.5 * (L_im + R_im);
            spec_LFE[k][0] = mono_re * lfe_weights[k];
            spec_LFE[k][1] = mono_im * lfe_weights[k];

            double TempL_re = L_re * hpf_weights[k];
            double TempL_im = L_im * hpf_weights[k];
            double TempR_re = R_re * hpf_weights[k];
            double TempR_im = R_im * hpf_weights[k];

            spec_TempL[k][0] = TempL_re;
            spec_TempL[k][1] = TempL_im;
            spec_TempR[k][0] = TempR_re;
            spec_TempR[k][1] = TempR_im;

            if (params.center_width >= 0.999) {
                raw_center_mask[k] = 1.0;
            }
            else {
                double P_L = TempL_re * TempL_re + TempL_im * TempL_im;
                double P_R = TempR_re * TempR_re + TempR_im * TempR_im;
                double mag_L = std::sqrt(P_L);
                double mag_R = std::sqrt(P_R);
                double pan = std::abs(mag_L - mag_R) / (mag_L + mag_R + eps);
                double cmask = 1.0 / (1.0 + std::pow(pan / effective_width, cslope_power));

                double dot = TempL_re * TempR_re + TempL_im * TempR_im;
                double icc = max(0.0, (2.0 * dot) / (P_L + P_R + eps));
                raw_center_mask[k] = cmask * icc;
            }
        }

        // Stage 1 Pass 2: Fast symmetric EMA temporal smoothing on Center mask
        constexpr double CENTER_SMOOTH_SEC = 0.005;
        const double center_alpha = std::exp(-(static_cast<double>(HOP_SIZE) / sample_rate) / CENTER_SMOOTH_SEC);
        for (size_t k = 0; k < spec_size; ++k) {
            double smoothed_c = center_alpha * prev_center_mask[k] + (1.0 - center_alpha) * raw_center_mask[k];
            prev_center_mask[k] = smoothed_c;
            spec_C[k][0] = 0.5 * (spec_TempL[k][0] + spec_TempR[k][0]) * smoothed_c;
            spec_C[k][1] = 0.5 * (spec_TempL[k][1] + spec_TempR[k][1]) * smoothed_c;
        }

        // Stage 1 Inverse FFTs & OLA
        fftw_execute(p_inv_LFE);
        fftw_execute(p_inv_TempL);
        fftw_execute(p_inv_TempR);
        fftw_execute(p_inv_C);

        synthesis_ola_avx(out_LFE, window.data(), ola_LFE.data(), ifft_scale, FFT_SIZE);
        synthesis_ola_avx(out_TempL, window.data(), ola_TempL.data(), ifft_scale, FFT_SIZE);
        synthesis_ola_avx(out_TempR, window.data(), ola_TempR.data(), ifft_scale, FFT_SIZE);
        synthesis_ola_avx(out_C, window.data(), ola_C.data(), ifft_scale, FFT_SIZE);

        float CVal = 0.0f, LVal = 0.0f, RVal = 0.0f, LFEVal = 0.0f;
        float LsVal = 0.0f, RsVal = 0.0f, LRVal = 0.0f, LRsVal = 0.0f, OutMax = 0.0f;

        if (warmup_blocks_1 > 0) {
            --warmup_blocks_1;
        }
        else {
            // ----------------------------------------------------------------
            // Center Downward Expanding Gate & Subtraction from TempL / TempR
            // ----------------------------------------------------------------
            if (!c_thresh_active) {
                std::copy(ola_C.begin(), ola_C.begin() + HOP_SIZE, gated_C.begin());
                subtract_avx(ola_TempL.data(), gated_C.data(), temp_L_postC.data(), HOP_SIZE);
                subtract_avx(ola_TempR.data(), gated_C.data(), temp_R_postC.data(), HOP_SIZE);
            }
            else {
                for (size_t i = 0; i < HOP_SIZE; ++i) {
                    double abs_C = std::abs(ola_C[i]);
                    if (abs_C > env_C) {
                        env_C = alpha_c_attack * env_C + (1.0 - alpha_c_attack) * abs_C;
                    }
                    else {
                        env_C = alpha_c_release * env_C + (1.0 - alpha_c_release) * abs_C;
                    }

                    double target_gain_C = compute_expander_gain(env_C, params.center_thresh_db, c_knee_high);
                    exp_gain_C = alpha_c_gain * exp_gain_C + (1.0 - alpha_c_gain) * target_gain_C;

                    gated_C[i] = ola_C[i] * exp_gain_C;
                    temp_L_postC[i] = ola_TempL[i] - gated_C[i];
                    temp_R_postC[i] = ola_TempR[i] - gated_C[i];
                }
            }

            // Delay LFE and gated Center to stay sample-aligned with Stage 2 STFT latency
            sync_delay_LFE.process(ola_LFE.data(), aligned_LFE.data(), HOP_SIZE);
            sync_delay_C.process(gated_C.data(), aligned_C.data(), HOP_SIZE);

            // ================================================================
            // STAGE 2: Ambient L/R Extraction from Post-Center TempL / TempR
            // ================================================================
            std::copy(delay2_L.begin() + HOP_SIZE, delay2_L.end(), delay2_L.begin());
            std::copy(delay2_R.begin() + HOP_SIZE, delay2_R.end(), delay2_R.begin());
            std::copy(temp_L_postC.begin(), temp_L_postC.end(), delay2_L.begin() + (FFT_SIZE - HOP_SIZE));
            std::copy(temp_R_postC.begin(), temp_R_postC.end(), delay2_R.begin() + (FFT_SIZE - HOP_SIZE));

            apply_window_avx(delay2_L.data(), window.data(), in2_L, FFT_SIZE);
            apply_window_avx(delay2_R.data(), window.data(), in2_R, FFT_SIZE);

            fftw_execute(p2_fwd_L);
            fftw_execute(p2_fwd_R);

            // Stage 2 Pass 1: Raw Ambient Mask from Post-Center Temporary L/R
            for (size_t k = 0; k < vec_bins; k += 4) {
                __m256d v_L_re, v_L_im, v_R_re, v_R_im;
                load_deinterleave_4complex(&spec2_L[k], v_L_re, v_L_im);
                load_deinterleave_4complex(&spec2_R[k], v_R_re, v_R_im);

                interleave_store_4complex(&spec_SurroundL[k], v_L_re, v_L_im);
                interleave_store_4complex(&spec_SurroundR[k], v_R_re, v_R_im);

                __m256d v_P_L = _mm256_fmadd_pd(v_L_re, v_L_re, _mm256_mul_pd(v_L_im, v_L_im));
                __m256d v_P_R = _mm256_fmadd_pd(v_R_re, v_R_re, _mm256_mul_pd(v_R_im, v_R_im));
                __m256d v_mag_L = _mm256_sqrt_pd(v_P_L);
                __m256d v_mag_R = _mm256_sqrt_pd(v_P_R);

                __m256d v_diff = _mm256_and_pd(_mm256_sub_pd(v_mag_L, v_mag_R), v_abs_mask);
                __m256d v_sum = _mm256_add_pd(_mm256_add_pd(v_mag_L, v_mag_R), v_eps);
                __m256d v_pan = _mm256_div_pd(v_diff, v_sum);

                __m256d v_dot = _mm256_fmadd_pd(v_L_re, v_R_re, _mm256_mul_pd(v_L_im, v_R_im));
                __m256d v_tot_pwr = _mm256_add_pd(_mm256_add_pd(v_P_L, v_P_R), v_eps);
                __m256d v_icc = _mm256_div_pd(_mm256_add_pd(v_dot, v_dot), v_tot_pwr);

                __m256d v_pos_icc = _mm256_max_pd(v_zero, v_icc);
                __m256d v_diff_icc = _mm256_max_pd(v_zero, _mm256_sub_pd(v_one, v_pos_icc));

                alignas(32) double r_pan[4];
                alignas(32) double r_diff[4];
                _mm256_storeu_pd(r_pan, v_pan);
                _mm256_storeu_pd(r_diff, v_diff_icc);

                for (int b = 0; b < 4; ++b) {
                    double pmask = 1.0;
                    if (params.surround_width < 0.999) {
                        double dist = (1.0 - r_pan[b]) / effective_surround_width;
                        pmask = (dist > 10.0) ? 0.0 : 1.0 / (1.0 + std::pow(dist, sslope_power));
                    }
                    raw_surround_mask[k + b] = r_diff[b] * pmask;
                }
            }

            // Stage 2 Pass 1 Scalar Tail
            for (size_t k = vec_bins; k < spec_size; ++k) {
                double L_re = spec2_L[k][0], L_im = spec2_L[k][1];
                double R_re = spec2_R[k][0], R_im = spec2_R[k][1];

                spec_SurroundL[k][0] = L_re;
                spec_SurroundL[k][1] = L_im;
                spec_SurroundR[k][0] = R_re;
                spec_SurroundR[k][1] = R_im;

                double P_L = L_re * L_re + L_im * L_im;
                double P_R = R_re * R_re + R_im * R_im;
                double mag_L = std::sqrt(P_L);
                double mag_R = std::sqrt(P_R);

                double pan = std::abs(mag_L - mag_R) / (mag_L + mag_R + eps);
                double dot = L_re * R_re + L_im * R_im;
                double icc = (2.0 * dot) / (P_L + P_R + eps);
                double diff_icc = max(0.0, 1.0 - max(0.0, icc));

                double pmask = 1.0;
                if (params.surround_width < 0.999) {
                    double dist = (1.0 - pan) / effective_surround_width;
                    pmask = (dist > 10.0) ? 0.0 : 1.0 / (1.0 + std::pow(dist, sslope_power));
                }
                raw_surround_mask[k] = diff_icc * pmask;
            }

            // Stage 2 Pass 2: Spectral Smoothing on Surround Mask
            spec_smoothed_mask[0] = 0.75 * raw_surround_mask[0] + 0.25 * raw_surround_mask[1];
            for (size_t k = 1; k < spec_size - 1; ++k) {
                spec_smoothed_mask[k] = 0.25 * raw_surround_mask[k - 1] + 0.50 * raw_surround_mask[k] + 0.25 * raw_surround_mask[k + 1];
            }
            spec_smoothed_mask[spec_size - 1] = 0.75 * raw_surround_mask[spec_size - 1] + 0.25 * raw_surround_mask[spec_size - 2];

            // Stage 2 Pass 3: Temporal Smoothing & Apply Surround Mask
            for (size_t k = 0; k < spec_size; ++k) {
                double smoothed = temporal_alpha * prev_surround_mask[k] + (1.0 - temporal_alpha) * spec_smoothed_mask[k];
                prev_surround_mask[k] = smoothed;

                spec_SurroundL[k][0] *= smoothed;
                spec_SurroundL[k][1] *= smoothed;
                spec_SurroundR[k][0] *= smoothed;
                spec_SurroundR[k][1] *= smoothed;
            }

            // Stage 2 Inverse FFTs & OLA
            fftw_execute(p_inv_SurroundL);
            fftw_execute(p_inv_SurroundR);

            synthesis_ola_avx(out_SurroundL, window.data(), ola_SurroundL.data(), ifft_scale, FFT_SIZE);
            synthesis_ola_avx(out_SurroundR, window.data(), ola_SurroundR.data(), ifft_scale, FFT_SIZE);

            if (warmup_blocks_2 > 0) {
                --warmup_blocks_2;
            }
            else {
                // ------------------------------------------------------------
                // Surround Expander (optional), Subtraction from TempL/R, Delay, Gains
                // ------------------------------------------------------------
                if (!s_thresh_active) {
                    std::copy(ola_SurroundL.begin(), ola_SurroundL.begin() + HOP_SIZE, gated_SurroundL.begin());
                    std::copy(ola_SurroundR.begin(), ola_SurroundR.begin() + HOP_SIZE, gated_SurroundR.begin());
                }
                else {
                    for (size_t i = 0; i < HOP_SIZE; ++i) {
                        double abs_Ls = std::abs(ola_SurroundL[i]);
                        if (abs_Ls > env_Ls) {
                            env_Ls = alpha_s_attack * env_Ls + (1.0 - alpha_s_attack) * abs_Ls;
                        }
                        else {
                            env_Ls = alpha_s_release * env_Ls + (1.0 - alpha_s_release) * abs_Ls;
                        }
                        double target_gain_Ls = compute_expander_gain(env_Ls, params.surround_thresh_db, s_knee_high);
                        exp_gain_Ls = alpha_s_gain * exp_gain_Ls + (1.0 - alpha_s_gain) * target_gain_Ls;
                        gated_SurroundL[i] = ola_SurroundL[i] * exp_gain_Ls;

                        double abs_Rs = std::abs(ola_SurroundR[i]);
                        if (abs_Rs > env_Rs) {
                            env_Rs = alpha_s_attack * env_Rs + (1.0 - alpha_s_attack) * abs_Rs;
                        }
                        else {
                            env_Rs = alpha_s_release * env_Rs + (1.0 - alpha_s_release) * abs_Rs;
                        }
                        double target_gain_Rs = compute_expander_gain(env_Rs, params.surround_thresh_db, s_knee_high);
                        exp_gain_Rs = alpha_s_gain * exp_gain_Rs + (1.0 - alpha_s_gain) * target_gain_Rs;
                        gated_SurroundR[i] = ola_SurroundR[i] * exp_gain_Rs;
                    }
                }

                // Subtract surround L and R from temporary L and R respectively
                subtract_avx(delay2_L.data(), gated_SurroundL.data(), final_L_sub.data(), HOP_SIZE);
                subtract_avx(delay2_R.data(), gated_SurroundR.data(), final_R_sub.data(), HOP_SIZE);

                // Apply delay to surround L and R channels
                delay_line_Ls.process(gated_SurroundL.data(), delayed_SurroundL.data(), HOP_SIZE);
                delay_line_Rs.process(gated_SurroundR.data(), delayed_SurroundR.data(), HOP_SIZE);

                // Apply gains to C, LFE, Ls, Rs (L and R gains applied during interleaving)
                copy_with_gain_avx(aligned_C.data(), final_C.data(), g_lin_C, HOP_SIZE);
                copy_with_gain_avx(aligned_LFE.data(), final_LFE.data(), g_lin_LFE, HOP_SIZE);
                copy_with_gain_avx(delayed_SurroundL.data(), final_Ls.data(), g_lin_Ls, HOP_SIZE);
                copy_with_gain_avx(delayed_SurroundR.data(), final_Rs.data(), g_lin_Rs, HOP_SIZE);

                // Peak tracking, soft limiter, and 6-channel interleaving
                for (size_t i = 0; i < HOP_SIZE; ++i) {
                    CVal = std::max<float>(static_cast<float>(std::abs(final_C[i])), CValLast);
                    LVal = static_cast<float>(std::abs(final_L_sub[i]));
                    RVal = static_cast<float>(std::abs(final_R_sub[i]));
                    LFEVal = std::max<float>(static_cast<float>(std::abs(final_LFE[i])), LFEValLast);
                    LsVal = static_cast<float>(std::abs(final_Ls[i]));
                    RsVal = static_cast<float>(std::abs(final_Rs[i]));
                    LRVal = std::max<float>(std::max<float>(LVal, RVal), LRValLast);
                    LRsVal = std::max<float>(std::max<float>(LsVal, RsVal), LRsValLast);
                    CValLast = CVal;
                    LFEValLast = LFEVal;
                    LRValLast = LRVal;
                    LRsValLast = LRsVal;

                    float maxVal = static_cast<float>(std::abs(final_C[i]));
                    maxVal = std::max<float>(maxVal, static_cast<float>(std::abs(final_L_sub[i]) * g_lin_L));
                    maxVal = std::max<float>(maxVal, static_cast<float>(std::abs(final_R_sub[i]) * g_lin_R));
                    maxVal = std::max<float>(maxVal, static_cast<float>(std::abs(final_Ls[i])));
                    maxVal = std::max<float>(maxVal, static_cast<float>(std::abs(final_Rs[i])));
                    maxVal = std::max<float>(maxVal, static_cast<float>(std::abs(final_LFE[i])));

                    if (use_limiter && maxVal * currentLimiterGain > limiter_threshold) {
                        currentLimiterGain = limiter_threshold / (maxVal + 1e-6f);
                    }
                    maxVal *= currentLimiterGain;
                    maxVal = std::max<float>(maxVal, MaxValLast);
                    MaxValLast = maxVal;
                    OutMax = maxVal;

                    interleaved_out[i * 6 + 0] = final_L_sub[i] * currentLimiterGain * g_lin_L;
                    interleaved_out[i * 6 + 1] = final_R_sub[i] * currentLimiterGain * g_lin_R;
                    interleaved_out[i * 6 + 2] = final_C[i] * currentLimiterGain;
                    interleaved_out[i * 6 + 3] = final_LFE[i] * currentLimiterGain;
                    interleaved_out[i * 6 + 4] = final_Ls[i] * currentLimiterGain;
                    interleaved_out[i * 6 + 5] = final_Rs[i] * currentLimiterGain;

                    if (currentLimiterGain < 1.0f) {
                        currentLimiterGain += (1.0f - currentLimiterGain) * (1.0f - RECOVERY_COEFF);
                    }
                }

                // Convert interleaved double 5.1 to target encoder format & push to out_fifo
                const uint8_t* in_ptrs_resamp[1] = { reinterpret_cast<const uint8_t*>(interleaved_out.data()) };
                int out_samples = swr_convert(swr_out, converted_resamp_buf, max_resampled_samples,
                    in_ptrs_resamp, static_cast<int>(HOP_SIZE));
                if (out_samples > 0) {
                    av_audio_fifo_write(out_fifo, reinterpret_cast<void**>(converted_resamp_buf), out_samples);
                }

                while (av_audio_fifo_size(out_fifo) >= enc_frame_size) {
                    AVFrame* frame = av_frame_alloc();
                    frame->nb_samples = enc_frame_size;
                    frame->format = enc_ctx->sample_fmt;
                    frame->sample_rate = enc_ctx->sample_rate;
#if USE_NEW_CH_LAYOUT
                    av_channel_layout_copy(&frame->ch_layout, &enc_ctx->ch_layout);
#else
                    frame->channels = enc_ctx->channels;
                    frame->channel_layout = enc_ctx->channel_layout;
#endif
                    av_frame_get_buffer(frame, 0);

                    av_audio_fifo_read(out_fifo, reinterpret_cast<void**>(frame->data), enc_frame_size);
                    frame->pts = pts_counter;
                    pts_counter += enc_frame_size;

                    encode_and_write_frame(enc_ctx, fmt_ctx_out, out_stream, frame);
                    av_frame_free(&frame);
                }
            }

            // Shift Stage 2 OLA accumulators
            shift_ola(ola_SurroundL);
            shift_ola(ola_SurroundR);
        }

        // Shift Stage 1 OLA accumulators
        shift_ola(ola_LFE);
        shift_ola(ola_TempL);
        shift_ola(ola_TempR);
        shift_ola(ola_C);

        framesProcessed += samples_to_read;
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::milliseconds>(now - last_update).count() > 250 || (totalFrames > 0 && framesProcessed >= totalFrames)) {
            double percent = (totalFrames > 0) ? min(100.0, 100.0 * framesProcessed / totalFrames) : 0.0;
            std::cout << "\rProcessing: " << "(" << static_cast<int>(percent)
                << "%) completed - Output Max (Max) Channel Max (L/R) (C) (LFE) (Ls/Rs): "
                << std::fixed << std::setprecision(2) << OutMax << " " << LRVal << " "
                << CVal << " " << LFEVal << " " << LRsVal << std::flush;
            last_update = now;
        }
    }

    // ------------------------------------------------------------------------
    // 6. Flush Remainder, Encoders, & Muxer
    // ------------------------------------------------------------------------
    int flush_samples = 0;
    while ((flush_samples = swr_convert(swr_out, converted_resamp_buf, max_resampled_samples, nullptr, 0)) > 0) {
        av_audio_fifo_write(out_fifo, reinterpret_cast<void**>(converted_resamp_buf), flush_samples);
    }

    const bool allow_small_last = (enc_codec->capabilities & (AV_CODEC_CAP_VARIABLE_FRAME_SIZE | AV_CODEC_CAP_SMALL_LAST_FRAME)) != 0;
    while (av_audio_fifo_size(out_fifo) > 0) {
        int rem = av_audio_fifo_size(out_fifo);
        int to_read = min(rem, enc_frame_size);
        int frame_sz = allow_small_last ? to_read : enc_frame_size;

        AVFrame* frame = av_frame_alloc();
        frame->nb_samples = frame_sz;
        frame->format = enc_ctx->sample_fmt;
        frame->sample_rate = enc_ctx->sample_rate;
#if USE_NEW_CH_LAYOUT
        av_channel_layout_copy(&frame->ch_layout, &enc_ctx->ch_layout);
        av_frame_get_buffer(frame, 0);
        if (to_read < frame_sz) {
            av_samples_set_silence(frame->data, 0, frame_sz, enc_ctx->ch_layout.nb_channels, enc_ctx->sample_fmt);
        }
#else
        frame->channels = enc_ctx->channels;
        frame->channel_layout = enc_ctx->channel_layout;
        av_frame_get_buffer(frame, 0);
        if (to_read < frame_sz) {
            av_samples_set_silence(frame->data, 0, frame_sz, enc_ctx->channels, enc_ctx->sample_fmt);
        }
#endif

        av_audio_fifo_read(out_fifo, reinterpret_cast<void**>(frame->data), to_read);

        frame->pts = pts_counter;
        pts_counter += frame_sz;

        encode_and_write_frame(enc_ctx, fmt_ctx_out, out_stream, frame);
        av_frame_free(&frame);
    }

    encode_and_write_frame(enc_ctx, fmt_ctx_out, out_stream, nullptr);
    av_write_trailer(fmt_ctx_out);

    // ------------------------------------------------------------------------
    // 7. Cleanup
    // ------------------------------------------------------------------------
    av_freep(&dbl_tmp);
    if (converted_resamp_buf) {
        av_freep(&converted_resamp_buf[0]);
        av_freep(&converted_resamp_buf);
    }
    av_frame_free(&dec_frame);
    av_packet_free(&in_pkt);
    av_audio_fifo_free(in_fifo);
    av_audio_fifo_free(out_fifo);
    swr_free(&swr_in);
    swr_free(&swr_out);

    fftw_destroy_plan(p_fwd_L);
    fftw_destroy_plan(p_fwd_R);
    fftw_destroy_plan(p_inv_LFE);
    fftw_destroy_plan(p_inv_TempL);
    fftw_destroy_plan(p_inv_TempR);
    fftw_destroy_plan(p_inv_C);

    fftw_destroy_plan(p2_fwd_L);
    fftw_destroy_plan(p2_fwd_R);
    fftw_destroy_plan(p_inv_SurroundL);
    fftw_destroy_plan(p_inv_SurroundR);

    fftw_free(in_L);      fftw_free(in_R);
    fftw_free(spec_L);    fftw_free(spec_R);
    fftw_free(spec_LFE);  fftw_free(spec_TempL);
    fftw_free(spec_TempR); fftw_free(spec_C);
    fftw_free(out_LFE);   fftw_free(out_TempL);
    fftw_free(out_TempR); fftw_free(out_C);

    fftw_free(in2_L);     fftw_free(in2_R);
    fftw_free(spec2_L);   fftw_free(spec2_R);
    fftw_free(spec_SurroundL); fftw_free(spec_SurroundR);
    fftw_free(out_SurroundL);  fftw_free(out_SurroundR);

    avcodec_free_context(&dec_ctx);
    avcodec_free_context(&enc_ctx);
    avformat_close_input(&fmt_ctx_in);
    if (!(fmt_ctx_out->oformat->flags & AVFMT_NOFILE)) {
        avio_closep(&fmt_ctx_out->pb);
    }
    avformat_free_context(fmt_ctx_out);

    return true;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        printUsage();
        return 1;
    }
    const char* in_path = nullptr;
    const char* out_surround_path = nullptr;

    SurroundParams params;
    try {
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if ((arg == "--input" || arg == "-i") && i + 1 < argc) in_path = argv[++i];
            else if ((arg == "--output" || arg == "-o") && i + 1 < argc) out_surround_path = argv[++i];
            else if ((arg == "--cwidth" || arg == "-cw") && i + 1 < argc) params.center_width = std::stod(argv[++i]);
            else if ((arg == "--cslope" || arg == "-cs") && i + 1 < argc) params.center_slope = std::stod(argv[++i]);
            else if ((arg == "--lfeclp" || arg == "-lp" || arg == "--lfecutoff" || arg == "-lc") && i + 1 < argc) params.lfe_cutoff_hz = std::stod(argv[++i]);
            else if ((arg == "--swidth" || arg == "-sw") && i + 1 < argc) params.surround_width = std::stod(argv[++i]);
            else if ((arg == "--sslope" || arg == "-ss") && i + 1 < argc) params.surround_slope = std::stod(argv[++i]);
            else if ((arg == "--sdelay" || arg == "-sd") && i + 1 < argc) params.surround_delay_ms = std::stod(argv[++i]);
            else if ((arg == "--gainlr" || arg == "-gf") && i + 1 < argc) params.gain_L_db = std::stod(argv[++i]);
            else if ((arg == "--gainc" || arg == "-gc") && i + 1 < argc) params.gain_C_db = std::stod(argv[++i]);
            else if ((arg == "--gainlfe" || arg == "-gl") && i + 1 < argc) params.gain_lfe_db = std::stod(argv[++i]);
            else if ((arg == "--gainslsr" || arg == "-gs") && i + 1 < argc) params.gain_Ls_db = std::stod(argv[++i]);
            else if ((arg == "--cthresh" || arg == "-ct") && i + 1 < argc) params.center_thresh_db = std::stod(argv[++i]);
            else if ((arg == "--sthresh" || arg == "-st") && i + 1 < argc) params.surround_thresh_db = std::stod(argv[++i]);
            else if ((arg == "--format" || arg == "-f") && i + 1 < argc) params.out_format = argv[++i];
            else if ((arg == "--codec" || arg == "-c") && i + 1 < argc) params.out_codec = argv[++i];
            else if ((arg == "--samplerate" || arg == "-sr") && i + 1 < argc) {
                params.out_sample_rate = std::stoi(argv[++i]);
                if (params.out_sample_rate <= 0) {
                    std::cerr << "Error: --samplerate must be a positive integer." << std::endl;
                    return 1;
                }
            }
            else if ((arg == "--samplefmt" || arg == "-sf") && i + 1 < argc) params.out_sample_fmt = argv[++i];
            else if ((arg == "--bits" || arg == "-b") && i + 1 < argc) {
                params.out_bits_per_sample = std::stoi(argv[++i]);
                if (params.out_bits_per_sample != 16 && params.out_bits_per_sample != 24 && params.out_bits_per_sample != 32) {
                    std::cerr << "Error: --bits must be 16, 24, or 32." << std::endl;
                    return 1;
                }
            }
            else if ((arg == "--bitrate" || arg == "-br") && i + 1 < argc) {
                std::string br_str = argv[++i];
                if (!parse_bitrate(br_str, params.out_bitrate)) {
                    std::cerr << "Error: Invalid --bitrate value '" << br_str << "'." << std::endl;
                    return 1;
                }
            }
            else if ((arg == "--compression" || arg == "-cl") && i + 1 < argc) {
                params.out_compression_level = std::stoi(argv[++i]);
                if (params.out_compression_level < 0) {
                    std::cerr << "Error: --compression must be >= 0." << std::endl;
                    return 1;
                }
            }
            else if ((arg == "--chlayout" || arg == "-ch") && i + 1 < argc) params.out_ch_layout = argv[++i];
            else if ((arg == "--limiter" || arg == "-lt") && i + 1 < argc) {
                std::string lim_str = to_lower_ascii(argv[++i]);
                params.limiter_override = true;
                if (lim_str == "off" || lim_str == "none" || lim_str == "0") {
                    params.limiter_enabled = false;
                }
                else {
                    params.limiter_enabled = true;
                    params.limiter_threshold = std::stof(lim_str);
                    if (params.limiter_threshold <= 0.0f) {
                        std::cerr << "Error: --limiter threshold must be > 0 or 'off'." << std::endl;
                        return 1;
                    }
                }
            }
            else if ((arg == "--encopt" || arg == "-eo") && i + 1 < argc) {
                std::string kv = argv[++i];
                size_t eq = kv.find('=');
                if (eq == std::string::npos || eq == 0 || eq + 1 >= kv.size()) {
                    std::cerr << "Error: --encopt requires key=value format (got '" << kv << "')." << std::endl;
                    return 1;
                }
                params.out_enc_options.emplace_back(kv.substr(0, eq), kv.substr(eq + 1));
            }
            else if (arg == "--help" || arg == "-h") {
                printUsage();
                return 0;
            }
            else {
                std::cerr << "Error: Unknown or incomplete argument '" << arg << "'." << std::endl;
                printUsage();
                return 1;
            }
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Error parsing arguments: " << e.what() << std::endl;
        return 1;
    }

    params.gain_R_db = params.gain_L_db;
    params.gain_Rs_db = params.gain_Ls_db;

    if (!in_path || !out_surround_path) {
        std::cerr << "Error: Input and output paths are required." << std::endl;
        printUsage();
        return 1;
    }

    std::cout << "Processing Audio with FFmpeg & 4-Bin AVX SIMD Acceleration..." << std::endl;
    std::cout << " - Center Width       : " << params.center_width << std::endl;
    std::cout << " - Center Slope       : " << params.center_slope << std::endl;
    std::cout << " - Surround Width      : " << params.surround_width
        << (params.surround_width >= 0.999 ? " (entire L/R channels)" : " (restricted to extreme pans)") << std::endl;
    std::cout << " - Surround Slope      : " << params.surround_slope << std::endl;
    std::cout << " - Surround Delay      : " << params.surround_delay_ms << " ms" << std::endl;
    std::cout << " - LFE Cutoff         : " << params.lfe_cutoff_hz << " Hz" << std::endl;
    std::cout << " - Center Threshold   : " << params.center_thresh_db << " dB"
        << (params.center_thresh_db <= -95.0 ? " (off)" : "") << std::endl;
    std::cout << " - Surround Threshold  : " << params.surround_thresh_db << " dB"
        << (params.surround_thresh_db <= -95.0 ? " (off)" : "") << std::endl;
    std::cout << " - Gains (dB) [LR/C/LFE/LsRs]: "
        << params.gain_L_db << " / "
        << params.gain_C_db << " / "
        << params.gain_lfe_db << " / "
        << params.gain_Rs_db << " dB" << std::endl;
    std::cout << " - Input              : " << in_path << std::endl;
    std::cout << " - Output             : " << out_surround_path << std::endl;

    if (process_audio(in_path, out_surround_path, params)) {
        std::cout << "\nProcessing completed successfully." << std::endl;
    }
    else {
        std::cerr << "\nProcessing failed." << std::endl;
        return 1;
    }

    return 0;
}
