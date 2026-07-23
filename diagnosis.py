# def auto_diagnose(
#     freqs,
#     fft_amps,
#     bpfo,
#     bpfi,
#     threshold,
#     tolerance
# ):
#     mask = (
#         (freqs > 40.0) &
#         (freqs <= 500.0)
#     )

#     valid_freqs = freqs[mask]
#     valid_amps = fft_amps[mask]

#     max_idx = valid_amps.argmax()

#     f_max = valid_freqs[max_idx]
#     amp_max = valid_amps[max_idx]

#     print(
#         " MECHANICAL REASONING (PHYSICS VERIFICATION)"
#     )

#     if amp_max < threshold:
#         status = "NORMAL"

#         report_str = (
#             f"Diagnosis: Measured signal at {f_max:.2f} Hz "
#             f"has an amplitude ({amp_max:.4f}) below the warning threshold.\n"
#             f"Conclusion: No mechanical defects detected (Normal)."
#         )

#     else:
#         theoretical_targets = {
#             "BPFO": (
#                 abs(f_max - bpfo) / bpfo,
#                 bpfo,
#                 "Outer Ring Defect"
#             ),

#             "BPFI": (
#                 abs(f_max - bpfi) / bpfi,
#                 bpfi,
#                 "Inner Ring Defect"
#             ),
#         }

#         best_match_key = min(
#             theoretical_targets,
#             key=lambda k: theoretical_targets[k][0]
#         )

#         best_error, theo_freq, defect_name = (
#             theoretical_targets[best_match_key]
#         )

#         match_percent = (
#             1 - best_error
#         ) * 100

#         if best_error <= tolerance:
#             status = "RED ALERT"

#             report_str = (
#                 f"Diagnosis: Measured fault signal at {f_max:.2f} Hz "
#                 f"matches {match_percent:.1f}% with theoretical "
#                 f"{best_match_key} ({theo_freq:.2f} Hz).\n"
#                 f"Conclusion: {defect_name}."
#             )

#         else:
#             status = "YELLOW ALERT"

#             report_str = (
#                 f"Diagnosis: Measured signal at {f_max:.2f} Hz "
#                 f"does not strongly match theoretical bands "
#                 f"(Error > {tolerance*100:.0f}%).\n"
#                 f"Conclusion: Normal."
#             )

#     print(report_str)

#     return status, report_str
def auto_diagnose(
    freqs,
    fft_amps,
    bpfo,
    bpfi,
    threshold,
    tolerance
):
    mask = (
        (freqs > 40.0) &
        (freqs <= 500.0)
    )

    valid_freqs = freqs[mask]
    valid_amps = fft_amps[mask]

    max_idx = valid_amps.argmax()

    f_max = valid_freqs[max_idx]
    amp_max = valid_amps[max_idx]

    print(
        " MECHANICAL REASONING (PHYSICS VERIFICATION)"
    )

    if amp_max < threshold:
        status = "NORMAL"

        report_str = (
            f"Diagnosis: Measured signal at {f_max:.2f} Hz "
            f"has an amplitude ({amp_max:.4f}) below the warning threshold.\n"
            f"Conclusion: No mechanical defects detected (Normal)."
        )

    else:
        theoretical_targets = {
            "BPFO": (
                abs(f_max - bpfo) / bpfo,
                bpfo,
                "Outer Ring Defect"
            ),

            "BPFI": (
                abs(f_max - bpfi) / bpfi,
                bpfi,
                "Inner Ring Defect"
            ),
        }

        best_match_key = min(
            theoretical_targets,
            key=lambda k: theoretical_targets[k][0]
        )

        best_error, theo_freq, defect_name = (
            theoretical_targets[best_match_key]
        )

        match_percent = (
            1 - best_error
        ) * 100

        if best_error <= tolerance:
            status = "RED ALERT"

            report_str = (
                f"Diagnosis: Measured fault signal at {f_max:.2f} Hz "
                f"matches {match_percent:.1f}% with theoretical "
                f"{best_match_key} ({theo_freq:.2f} Hz).\n"
                f"Conclusion: {defect_name}."
            )

        else:
            # CHUYỂN ĐỔI TỪ YELLOW ALERT THÀNH NORMAL (MÀU XANH)
            status = "NORMAL"

            report_str = (
                f"Diagnosis: Measured signal at {f_max:.2f} Hz "
                f"exhibits high amplitude ({amp_max:.4f}) but does not strongly match theoretical bands "
                f"(Error {best_error*100:.1f}% > {tolerance*100:.0f}%).\n"
                f"Conclusion: Normal baseline fluctuation."
            )

    print(report_str)
    return status, report_str