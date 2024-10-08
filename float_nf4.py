import torch

def calc_mantissa(abs_x, exponent, normal_mask, MANTISSA_BITS, EXPONENT_BIAS, generator=None):
    mantissa_scaled = torch.where(
        normal_mask,
        (abs_x / (2.0 ** (exponent - EXPONENT_BIAS)) - 1.0) * (2**MANTISSA_BITS),
        (abs_x / (2.0 ** (-EXPONENT_BIAS + 1 - MANTISSA_BITS)))
    )

    mantissa_scaled += torch.rand(mantissa_scaled.size(), dtype=mantissa_scaled.dtype, layout=mantissa_scaled.layout, device=mantissa_scaled.device, generator=generator)
    return mantissa_scaled.floor() / (2**MANTISSA_BITS)

def manual_stochastic_round_to_nf4(x, generator=None):
    '''
    if dtype == torch.fp4_e2m1:
        EXPONENT_BITS, MANTISSA_BITS, EXPONENT_BIAS = 2, 1, 3
    elif dtype == torch.fp4_e3m0:
        EXPONENT_BITS, MANTISSA_BITS, EXPONENT_BIAS = 3, 0, 7
    '''
    EXPONENT_BITS, MANTISSA_BITS, EXPONENT_BIAS = 2, 1, 7
    # EXPONENT_BITS, MANTISSA_BITS, EXPONENT_BIAS = 3, 0, 15

    x = x.half()
    sign = torch.sign(x)
    abs_x = x.abs()
    sign = torch.where(abs_x == 0, 0, sign)

    # Combine exponent calculation and clamping
    exponent = torch.clamp(
        torch.floor(torch.log2(abs_x)) + EXPONENT_BIAS,
        0, 2**EXPONENT_BITS - 1
    )

    # Combine mantissa calculation and rounding
    normal_mask = ~(exponent == 0)

    abs_x[:] = calc_mantissa(abs_x, exponent, normal_mask, MANTISSA_BITS, EXPONENT_BIAS, generator=generator)

    sign *= torch.where(
        normal_mask,
        (2.0 ** (exponent - EXPONENT_BIAS)) * (1.0 + abs_x),
        (2.0 ** (-EXPONENT_BIAS + 1)) * abs_x
    )

    return sign

def stochastic_rounding_nf4(value, seed=0):
    generator = torch.Generator(device=value.device)
    generator.manual_seed(seed)
    output = torch.empty_like(value, dtype=torch.float16)
    num_slices = max(1, (value.numel() / (4096 * 4096)))
    slice_size = max(1, round(value.shape[0] / num_slices))
    for i in range(0, value.shape[0], slice_size):
        output[i:i+slice_size].copy_(manual_stochastic_round_to_nf4(value[i:i+slice_size], generator=generator))
    return output
