import torch
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms as T
from PIL import Image

from .constants import (CLIP_MEAN, CLIP_STD, IMAGENET_MEAN, IMAGENET_STD,
                        SIGLIP_MEAN, SIGLIP_STD)


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

    
def build_transform(input_size, pad2square=False, normalize_type='imagenet'):
    if normalize_type == 'imagenet':
        MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    elif normalize_type == 'clip':
        MEAN, STD = CLIP_MEAN, CLIP_STD
    elif normalize_type == 'siglip':
        MEAN, STD = SIGLIP_MEAN, SIGLIP_STD
    else:
        raise NotImplementedError

    if isinstance(input_size,int):
        final_input_size0 = final_input_size1 = input_size
    else:
        final_input_size0 = input_size[0]
        final_input_size1 = input_size[1]
    if pad2square is False:  # now we use this transform function by default
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((final_input_size0, final_input_size1), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
    else:
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Lambda(lambda img: expand2square(img, tuple(int(x * 255) for x in MEAN))),
            T.Resize((final_input_size0, final_input_size1), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
            ])

    return transform


@torch.no_grad()
def print_module_summary(module, max_nesting=1, skip_redundant=True):
    assert isinstance(module, torch.nn.Module)
    assert not isinstance(module, torch.jit.ScriptModule)

    header = ['Module Name', 'Total Params', 'Learnable Params']
    
    learnable_module_rows = []
    non_learnable_module_rows = []

    overall_total_param_count = sum(p.numel() for p in module.parameters())
    overall_learnable_param_count = sum(p.numel() for p in module.parameters() if p.requires_grad)
    # overall_buffer_count is no longer displayed but might be needed for skip_redundant if it used direct_buffer_size

    for name, mod_instance in module.named_modules():
        # Filter by nesting depth
        if name:  # Not the top-level module itself
            depth = name.count('.')
            if depth >= max_nesting:
                continue

        # Parameters and buffers *directly* in this module (for skip_redundant logic)
        mod_direct_params_list = list(mod_instance.parameters(recurse=False))
        mod_direct_buffers_list = list(mod_instance.buffers(recurse=False)) # Still needed for skip_redundant

        # Skip redundant modules: not top-level, no direct params/buffers, and no children
        if skip_redundant and mod_instance is not module and \
           not mod_direct_params_list and not mod_direct_buffers_list and \
           not list(mod_instance.children()):
            continue
            
        # Recursive parameter counts for display
        module_total_param_size = sum(p.numel() for p in mod_instance.parameters(recurse=True))
        module_learnable_param_size = sum(p.numel() for p in mod_instance.parameters(recurse=True) if p.requires_grad)
        
        display_name = name if name else type(module).__name__
        
        row_data = [
            display_name,
            str(module_total_param_size) if module_total_param_size > 0 else '-',
            str(module_learnable_param_size) if module_learnable_param_size > 0 else '-'
        ]

        if module_learnable_param_size > 0:
            learnable_module_rows.append(row_data)
        else:
            non_learnable_module_rows.append(row_data)

    # Assemble the final list of rows for printing
    final_rows = [header]
    final_rows.append(['---'] * len(header))

    if learnable_module_rows:
        final_rows.append(['Learnable Modules:', '', '']) # Section header
        final_rows.extend(learnable_module_rows)
        final_rows.append(['---'] * len(header))
    
    if non_learnable_module_rows:
        final_rows.append(['Non-Learnable Modules:', '', '']) # Section header
        final_rows.extend(non_learnable_module_rows)
        final_rows.append(['---'] * len(header))

    final_rows.append(['Total (Overall Model)', str(overall_total_param_count), str(overall_learnable_param_count)])

    # Print table
    if len(final_rows) <= 2: # Only header and one separator
        print("No modules to display.")
        return

    # Calculate column widths
    widths = [0] * len(header)
    for row in final_rows:
        # Ensure row has the same number of columns as header for width calculation
        # Section headers might have fewer "data" cells but should align
        for i, cell in enumerate(row):
            if i < len(widths): # Only process up to the number of columns in the header
                 widths[i] = max(widths[i], len(str(cell)))
            
    print()
    for row in final_rows:
        # Adjust printing for rows that might not have all columns (like section headers)
        formatted_row = []
        for i, cell in enumerate(row):
            if i < len(widths):
                formatted_row.append(str(cell).ljust(widths[i]))
            else: # Should not happen if rows are structured correctly
                formatted_row.append(str(cell))
        print('  '.join(formatted_row))
    print()