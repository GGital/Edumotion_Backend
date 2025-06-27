import numpy as np

def interpolate_missing_frames(sequence, frame_indices):
    full_sequence = []
    n_frames = frame_indices[-1] + 1
    current_idx = 0

    for i in range(n_frames):
        if i in frame_indices:
            full_sequence.append(sequence[current_idx])
            current_idx += 1
        else:
            # Interpolate linearly
            prev_idx = max([j for j in frame_indices if j < i], default=None)
            next_idx = min([j for j in frame_indices if j > i], default=None)
            if prev_idx is not None and next_idx is not None:
                ratio = (i - prev_idx) / (next_idx - prev_idx)
                interp = (
                    (1 - ratio) * sequence[frame_indices.index(prev_idx)] +
                    ratio * sequence[frame_indices.index(next_idx)]
                )
                full_sequence.append(interp)
            elif prev_idx is not None:
                full_sequence.append(sequence[frame_indices.index(prev_idx)])
            elif next_idx is not None:
                full_sequence.append(sequence[frame_indices.index(next_idx)])
            else:
                full_sequence.append(np.zeros_like(sequence[0]))
    return np.array(full_sequence)
