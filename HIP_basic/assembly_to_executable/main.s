
# __CLANG_OFFLOAD_BUNDLE____START__ hip-amdgcn-amd-amdhsa--gfx906
	.amdgcn_target "amdgcn-amd-amdhsa--gfx906"
	.amdhsa_code_object_version 6
	.section	.text._Z20vector_square_kernelIfEvPT_S1_y,"axG",@progbits,_Z20vector_square_kernelIfEvPT_S1_y,comdat
	.protected	_Z20vector_square_kernelIfEvPT_S1_y ; -- Begin function _Z20vector_square_kernelIfEvPT_S1_y
	.globl	_Z20vector_square_kernelIfEvPT_S1_y
	.p2align	8
	.type	_Z20vector_square_kernelIfEvPT_S1_y,@function
_Z20vector_square_kernelIfEvPT_S1_y:    ; @_Z20vector_square_kernelIfEvPT_S1_y
; %bb.0:
	s_load_dword s0, s[4:5], 0x24
	s_load_dwordx2 s[8:9], s[4:5], 0x10
	s_add_u32 s10, s4, 24
	s_addc_u32 s11, s5, 0
	v_mov_b32_e32 v1, 0
	s_waitcnt lgkmcnt(0)
	s_and_b32 s12, s0, 0xffff
	s_mul_i32 s6, s6, s12
	v_add_u32_e32 v0, s6, v0
	v_cmp_gt_u64_e32 vcc, s[8:9], v[0:1]
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_3
; %bb.1:
	s_load_dword s13, s[10:11], 0x0
	s_load_dwordx4 s[0:3], s[4:5], 0x0
	s_mov_b32 s5, 0
	v_lshlrev_b64 v[2:3], 2, v[0:1]
	s_mov_b64 s[6:7], 0
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s4, s13, s12
	s_lshl_b64 s[10:11], s[4:5], 2
	v_mov_b32_e32 v4, s3
	v_mov_b32_e32 v5, s1
	v_mov_b32_e32 v6, s11
.LBB0_2:                                ; =>This Inner Loop Header: Depth=1
	v_add_co_u32_e32 v7, vcc, s2, v2
	v_addc_co_u32_e32 v8, vcc, v4, v3, vcc
	global_load_dword v9, v[7:8], off
	v_add_co_u32_e32 v7, vcc, s0, v2
	v_addc_co_u32_e32 v8, vcc, v5, v3, vcc
	v_add_co_u32_e32 v0, vcc, s4, v0
	v_addc_co_u32_e32 v1, vcc, 0, v1, vcc
	v_add_co_u32_e32 v2, vcc, s10, v2
	v_addc_co_u32_e32 v3, vcc, v3, v6, vcc
	v_cmp_le_u64_e32 vcc, s[8:9], v[0:1]
	s_or_b64 s[6:7], vcc, s[6:7]
	s_waitcnt vmcnt(0)
	v_mul_f32_e32 v9, v9, v9
	global_store_dword v[7:8], v9, off
	s_andn2_b64 exec, exec, s[6:7]
	s_cbranch_execnz .LBB0_2
.LBB0_3:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z20vector_square_kernelIfEvPT_S1_y
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 280
		.amdhsa_user_sgpr_count 6
		.amdhsa_user_sgpr_private_segment_buffer 1
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_flat_scratch_init 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_system_sgpr_private_segment_wavefront_offset 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 10
		.amdhsa_next_free_sgpr 14
		.amdhsa_reserve_vcc 1
		.amdhsa_reserve_flat_scratch 0
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text._Z20vector_square_kernelIfEvPT_S1_y,"axG",@progbits,_Z20vector_square_kernelIfEvPT_S1_y,comdat
.Lfunc_end0:
	.size	_Z20vector_square_kernelIfEvPT_S1_y, .Lfunc_end0-_Z20vector_square_kernelIfEvPT_S1_y
                                        ; -- End function
	.set _Z20vector_square_kernelIfEvPT_S1_y.num_vgpr, 10
	.set _Z20vector_square_kernelIfEvPT_S1_y.num_agpr, 0
	.set _Z20vector_square_kernelIfEvPT_S1_y.numbered_sgpr, 14
	.set _Z20vector_square_kernelIfEvPT_S1_y.num_named_barrier, 0
	.set _Z20vector_square_kernelIfEvPT_S1_y.private_seg_size, 0
	.set _Z20vector_square_kernelIfEvPT_S1_y.uses_vcc, 1
	.set _Z20vector_square_kernelIfEvPT_S1_y.uses_flat_scratch, 0
	.set _Z20vector_square_kernelIfEvPT_S1_y.has_dyn_sized_stack, 0
	.set _Z20vector_square_kernelIfEvPT_S1_y.has_recursion, 0
	.set _Z20vector_square_kernelIfEvPT_S1_y.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 192
; TotalNumSgprs: 18
; NumVgprs: 10
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 2
; VGPRBlocks: 2
; NumSGPRsForWavesPerEU: 18
; NumVGPRsForWavesPerEU: 10
; Occupancy: 10
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 6
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.section	.AMDGPU.csdata,"",@progbits
	.type	__hip_cuid_9355bdbd540905a,@object ; @__hip_cuid_9355bdbd540905a
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_9355bdbd540905a
__hip_cuid_9355bdbd540905a:
	.byte	0                               ; 0x0
	.size	__hip_cuid_9355bdbd540905a, 1

	.ident	"AMD clang version 22.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.2.0 26014 7b800a19466229b8479a78de19143dc33c3ab9b5)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_9355bdbd540905a
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .offset:         16
        .size:           8
        .value_kind:     by_value
      - .offset:         24
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         28
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         32
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         36
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         38
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         40
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         42
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         44
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         46
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         64
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         72
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         80
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         88
        .size:           2
        .value_kind:     hidden_grid_dims
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 280
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           _Z20vector_square_kernelIfEvPT_S1_y
    .private_segment_fixed_size: 0
    .sgpr_count:     18
    .sgpr_spill_count: 0
    .symbol:         _Z20vector_square_kernelIfEvPT_S1_y.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     10
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx906
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata

# __CLANG_OFFLOAD_BUNDLE____END__ hip-amdgcn-amd-amdhsa--gfx906

# __CLANG_OFFLOAD_BUNDLE____START__ host-x86_64-unknown-linux-gnu-
	.file	"main.hip"
                                        # Start of file scope inline assembly
	.globl	_ZSt21ios_base_library_initv

                                        # End of file scope inline assembly
	.section	.rodata.cst8,"aM",@progbits,8
	.p2align	3, 0x0                          # -- Begin function main
.LCPI0_0:
	.quad	0x401e848000000000              # double 7.62939453125
	.section	.rodata.cst16,"aM",@progbits,16
	.p2align	4, 0x0
.LCPI0_1:
	.quad	2                               # 0x2
	.quad	3                               # 0x3
.LCPI0_2:
	.byte	0                               # 0x0
	.byte	0                               # 0x0
	.byte	0                               # 0x0
	.byte	0                               # 0x0
	.byte	0                               # 0x0
	.byte	0                               # 0x0
	.byte	0                               # 0x0
	.byte	0                               # 0x0
	.byte	1                               # 0x1
	.byte	0                               # 0x0
	.byte	0                               # 0x0
	.byte	0                               # 0x0
	.byte	0                               # 0x0
	.byte	0                               # 0x0
	.byte	0                               # 0x0
	.byte	0                               # 0x0
.LCPI0_3:
	.long	0x3fcf1aa0                      # float 1.61800003
	.long	0x3fcf1aa0                      # float 1.61800003
	.long	0x3fcf1aa0                      # float 1.61800003
	.long	0x3fcf1aa0                      # float 1.61800003
.LCPI0_4:
	.quad	4                               # 0x4
	.quad	4                               # 0x4
.LCPI0_5:
	.quad	8                               # 0x8
	.quad	8                               # 0x8
	.text
	.globl	main
	.p2align	4
	.type	main,@function
main:                                   # @main
.Lfunc_begin0:
	.cfi_startproc
	.cfi_personality 3, __gxx_personality_v0
	.cfi_lsda 3, .Lexception0
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%r12
	.cfi_def_cfa_offset 40
	pushq	%rbx
	.cfi_def_cfa_offset 48
	subq	$1600, %rsp                     # imm = 0x640
	.cfi_def_cfa_offset 1648
	.cfi_offset %rbx, -48
	.cfi_offset %r12, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	.cfi_escape 0x2e, 0x00
	leaq	128(%rsp), %rdi
	xorl	%esi, %esi
	callq	hipGetDevicePropertiesR0600
	testl	%eax, %eax
	jne	.LBB0_122
# %bb.1:
	.cfi_escape 0x2e, 0x00
	movl	$_ZSt4cout, %edi
	movl	$.L.str.3, %esi
	movl	$23, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	.cfi_escape 0x2e, 0x00
	leaq	128(%rsp), %rbx
	movq	%rbx, %rdi
	callq	strlen
	.cfi_escape 0x2e, 0x00
	movl	$_ZSt4cout, %edi
	movq	%rbx, %rsi
	movq	%rax, %rdx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	.cfi_escape 0x2e, 0x00
	movl	$_ZSt4cout, %edi
	movl	$.L.str.4, %esi
	movl	$1, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	.cfi_escape 0x2e, 0x00
	movl	$_ZSt4cout, %edi
	movl	$.L.str.5, %esi
	movl	$25, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	.cfi_escape 0x2e, 0x00
	movq	.LCPI0_0(%rip), %xmm0           # xmm0 = [7.62939453125E+0,0.0E+0]
	movl	$_ZSt4cout, %edi
	callq	_ZNSo9_M_insertIdEERSoT_
	movq	%rax, %rbx
	.cfi_escape 0x2e, 0x00
	movl	$.L.str.6, %esi
	movl	$5, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	.cfi_escape 0x2e, 0x00
	movl	$.L.str.4, %esi
	movl	$1, %edx
	movq	%rbx, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	.cfi_escape 0x2e, 0x00
	movl	$4000000, %edi                  # imm = 0x3D0900
	callq	_Znwm
	movq	%rax, %rbx
	.cfi_escape 0x2e, 0x00
	movl	$4000000, %edx                  # imm = 0x3D0900
	movq	%rax, %rdi
	xorl	%esi, %esi
	callq	memset@PLT
.Ltmp0:                                 # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	$4000000, %edi                  # imm = 0x3D0900
	callq	_Znwm
.Ltmp1:                                 # EH_LABEL
# %bb.2:
	movq	%rax, %r14
	.cfi_escape 0x2e, 0x00
	movl	$4000000, %edx                  # imm = 0x3D0900
	movq	%rax, %rdi
	xorl	%esi, %esi
	callq	memset@PLT
	movdqa	.LCPI0_1(%rip), %xmm0           # xmm0 = [2,3]
	movdqa	.LCPI0_2(%rip), %xmm1           # xmm1 = [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0]
	movl	$4, %eax
	movaps	.LCPI0_3(%rip), %xmm2           # xmm2 = [1.61800003E+0,1.61800003E+0,1.61800003E+0,1.61800003E+0]
	movdqa	.LCPI0_4(%rip), %xmm3           # xmm3 = [4,4]
	movdqa	.LCPI0_5(%rip), %xmm4           # xmm4 = [8,8]
	.p2align	4
.LBB0_3:                                # =>This Inner Loop Header: Depth=1
	movq	%xmm1, %rcx
	xorps	%xmm5, %xmm5
	cvtsi2ss	%rcx, %xmm5
	pshufd	$238, %xmm1, %xmm6              # xmm6 = xmm1[2,3,2,3]
	movq	%xmm6, %rcx
	xorps	%xmm6, %xmm6
	cvtsi2ss	%rcx, %xmm6
	movq	%xmm0, %rcx
	xorps	%xmm7, %xmm7
	cvtsi2ss	%rcx, %xmm7
	pshufd	$238, %xmm0, %xmm8              # xmm8 = xmm0[2,3,2,3]
	movq	%xmm8, %rcx
	xorps	%xmm8, %xmm8
	cvtsi2ss	%rcx, %xmm8
	unpcklps	%xmm6, %xmm5                    # xmm5 = xmm5[0],xmm6[0],xmm5[1],xmm6[1]
	unpcklps	%xmm8, %xmm7                    # xmm7 = xmm7[0],xmm8[0],xmm7[1],xmm8[1]
	movlhps	%xmm7, %xmm5                    # xmm5 = xmm5[0],xmm7[0]
	addps	%xmm2, %xmm5
	movups	%xmm5, -16(%rbx,%rax,4)
	movdqa	%xmm1, %xmm5
	paddq	%xmm3, %xmm5
	movq	%xmm5, %rcx
	xorps	%xmm6, %xmm6
	cvtsi2ss	%rcx, %xmm6
	pshufd	$238, %xmm5, %xmm5              # xmm5 = xmm5[2,3,2,3]
	movq	%xmm5, %rcx
	xorps	%xmm5, %xmm5
	cvtsi2ss	%rcx, %xmm5
	movdqa	%xmm0, %xmm7
	paddq	%xmm3, %xmm7
	unpcklps	%xmm5, %xmm6                    # xmm6 = xmm6[0],xmm5[0],xmm6[1],xmm5[1]
	movq	%xmm7, %rcx
	xorps	%xmm5, %xmm5
	cvtsi2ss	%rcx, %xmm5
	pshufd	$238, %xmm7, %xmm7              # xmm7 = xmm7[2,3,2,3]
	movq	%xmm7, %rcx
	xorps	%xmm7, %xmm7
	cvtsi2ss	%rcx, %xmm7
	unpcklps	%xmm7, %xmm5                    # xmm5 = xmm5[0],xmm7[0],xmm5[1],xmm7[1]
	movlhps	%xmm5, %xmm6                    # xmm6 = xmm6[0],xmm5[0]
	addps	%xmm2, %xmm6
	movups	%xmm6, (%rbx,%rax,4)
	paddq	%xmm4, %xmm1
	paddq	%xmm4, %xmm0
	addq	$8, %rax
	cmpq	$1000004, %rax                  # imm = 0xF4244
	jne	.LBB0_3
# %bb.4:
.Ltmp3:                                 # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	$_ZSt4cout, %edi
	movl	$.L.str.7, %esi
	movl	$27, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp4:                                 # EH_LABEL
# %bb.5:
.Ltmp5:                                 # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	.LCPI0_0(%rip), %xmm0           # xmm0 = [7.62939453125E+0,0.0E+0]
	movl	$_ZSt4cout, %edi
	callq	_ZNSo9_M_insertIdEERSoT_
.Ltmp6:                                 # EH_LABEL
# %bb.6:
.Ltmp7:                                 # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	$.L.str.8, %esi
	movl	$5, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp8:                                 # EH_LABEL
# %bb.7:
.Ltmp9:                                 # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	16(%rsp), %rdi
	movl	$4000000, %esi                  # imm = 0x3D0900
	callq	hipMalloc
.Ltmp10:                                # EH_LABEL
# %bb.8:
	movl	%eax, %ebp
	testl	%eax, %eax
	jne	.LBB0_9
# %bb.19:
.Ltmp28:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	8(%rsp), %rdi
	movl	$4000000, %esi                  # imm = 0x3D0900
	callq	hipMalloc
.Ltmp29:                                # EH_LABEL
# %bb.20:
	movl	%eax, %ebp
	testl	%eax, %eax
	jne	.LBB0_21
# %bb.31:
.Ltmp47:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	$_ZSt4cout, %edi
	movl	$.L.str.9, %esi
	movl	$23, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp48:                                # EH_LABEL
# %bb.32:
	movq	16(%rsp), %rdi
.Ltmp50:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	$4000000, %edx                  # imm = 0x3D0900
	movq	%rbx, %rsi
	movl	$1, %ecx
	callq	hipMemcpy
.Ltmp51:                                # EH_LABEL
# %bb.33:
	movl	%eax, %ebp
	testl	%eax, %eax
	jne	.LBB0_34
# %bb.44:
.Ltmp69:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	$_ZSt4cout, %edi
	movl	$.L.str.10, %esi
	movl	$43, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp70:                                # EH_LABEL
# %bb.45:
.Ltmp71:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movabsq	$4294967808, %rdi               # imm = 0x100000200
	movabsq	$4294967552, %rdx               # imm = 0x100000100
	movl	$1, %esi
	movl	$1, %ecx
	xorl	%r8d, %r8d
	xorl	%r9d, %r9d
	callq	__hipPushCallConfiguration
.Ltmp72:                                # EH_LABEL
# %bb.46:
	testl	%eax, %eax
	jne	.LBB0_49
# %bb.47:
	movq	8(%rsp), %rax
	movq	16(%rsp), %rcx
	movq	%rax, 88(%rsp)
	movq	%rcx, 80(%rsp)
	movq	$1000000, 72(%rsp)              # imm = 0xF4240
	leaq	88(%rsp), %rax
	movq	%rax, 96(%rsp)
	leaq	80(%rsp), %rax
	movq	%rax, 104(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 112(%rsp)
.Ltmp73:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	leaq	56(%rsp), %rdi
	leaq	40(%rsp), %rsi
	leaq	32(%rsp), %rdx
	leaq	24(%rsp), %rcx
	callq	__hipPopCallConfiguration
.Ltmp74:                                # EH_LABEL
# %bb.48:
	movq	56(%rsp), %rsi
	movl	64(%rsp), %edx
	movq	40(%rsp), %rcx
	movl	48(%rsp), %r8d
.Ltmp75:                                # EH_LABEL
	.cfi_escape 0x2e, 0x10
	leaq	96(%rsp), %r9
	movl	$_Z20vector_square_kernelIfEvPT_S1_y, %edi
	pushq	24(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	40(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
.Ltmp76:                                # EH_LABEL
.LBB0_49:
.Ltmp77:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipGetLastError
.Ltmp78:                                # EH_LABEL
# %bb.50:
	movl	%eax, %ebp
	testl	%eax, %eax
	jne	.LBB0_51
# %bb.61:
.Ltmp96:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	$_ZSt4cout, %edi
	movl	$.L.str.11, %esi
	movl	$23, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp97:                                # EH_LABEL
# %bb.62:
	movq	8(%rsp), %rsi
.Ltmp98:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	$4000000, %edx                  # imm = 0x3D0900
	movq	%r14, %rdi
	movl	$2, %ecx
	callq	hipMemcpy
.Ltmp99:                                # EH_LABEL
# %bb.63:
	movl	%eax, %ebp
	testl	%eax, %eax
	jne	.LBB0_64
# %bb.74:
	movq	16(%rsp), %rdi
.Ltmp117:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipFree
.Ltmp118:                               # EH_LABEL
# %bb.75:
	movl	%eax, %ebp
	testl	%eax, %eax
	jne	.LBB0_76
# %bb.86:
	movq	8(%rsp), %rdi
.Ltmp136:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	callq	hipFree
.Ltmp137:                               # EH_LABEL
# %bb.87:
	movl	%eax, %ebp
	testl	%eax, %eax
	jne	.LBB0_88
# %bb.98:
.Ltmp155:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	$_ZSt4cout, %edi
	movl	$.L.str.12, %esi
	movl	$19, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp156:                               # EH_LABEL
# %bb.99:
	movl	$3, %r15d
	.p2align	4
.LBB0_100:                              # =>This Inner Loop Header: Depth=1
	movss	-12(%r14,%r15,4), %xmm0         # xmm0 = mem[0],zero,zero,zero
	movss	-12(%rbx,%r15,4), %xmm1         # xmm1 = mem[0],zero,zero,zero
	mulss	%xmm1, %xmm1
	ucomiss	%xmm1, %xmm0
	jne	.LBB0_101
	jp	.LBB0_101
# %bb.113:                              #   in Loop: Header=BB0_100 Depth=1
	movss	-8(%r14,%r15,4), %xmm0          # xmm0 = mem[0],zero,zero,zero
	movss	-8(%rbx,%r15,4), %xmm1          # xmm1 = mem[0],zero,zero,zero
	mulss	%xmm1, %xmm1
	ucomiss	%xmm1, %xmm0
	jne	.LBB0_102
	jp	.LBB0_102
# %bb.114:                              #   in Loop: Header=BB0_100 Depth=1
	movss	-4(%r14,%r15,4), %xmm0          # xmm0 = mem[0],zero,zero,zero
	movss	-4(%rbx,%r15,4), %xmm1          # xmm1 = mem[0],zero,zero,zero
	mulss	%xmm1, %xmm1
	ucomiss	%xmm1, %xmm0
	jne	.LBB0_103
	jp	.LBB0_103
# %bb.115:                              #   in Loop: Header=BB0_100 Depth=1
	movss	(%r14,%r15,4), %xmm0            # xmm0 = mem[0],zero,zero,zero
	movss	(%rbx,%r15,4), %xmm1            # xmm1 = mem[0],zero,zero,zero
	mulss	%xmm1, %xmm1
	ucomiss	%xmm1, %xmm0
	jne	.LBB0_104
	jp	.LBB0_104
# %bb.116:                              #   in Loop: Header=BB0_100 Depth=1
	addq	$4, %r15
	cmpq	$1000003, %r15                  # imm = 0xF4243
	jne	.LBB0_100
# %bb.117:
.Ltmp157:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	$_ZSt4cout, %edi
	movl	$.L.str.16, %esi
	movl	$8, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp158:                               # EH_LABEL
# %bb.118:
	.cfi_escape 0x2e, 0x00
	movl	$4000000, %esi                  # imm = 0x3D0900
	movq	%r14, %rdi
	callq	_ZdlPvm
	.cfi_escape 0x2e, 0x00
	movl	$4000000, %esi                  # imm = 0x3D0900
	movq	%rbx, %rdi
	callq	_ZdlPvm
	xorl	%eax, %eax
	addq	$1600, %rsp                     # imm = 0x640
	.cfi_def_cfa_offset 48
	popq	%rbx
	.cfi_def_cfa_offset 40
	popq	%r12
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	retq
.LBB0_102:
	.cfi_def_cfa_offset 1648
	addq	$-2, %r15
	jmp	.LBB0_104
.LBB0_103:
	decq	%r15
	jmp	.LBB0_104
.LBB0_101:
	addq	$-3, %r15
.LBB0_104:
.Ltmp160:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	$_ZSt4cerr, %edi
	movl	$.L.str.13, %esi
	movl	$15, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp161:                               # EH_LABEL
# %bb.105:
.Ltmp162:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	$_ZSt4cerr, %edi
	movq	%r15, %rsi
	callq	_ZNSo9_M_insertImEERSoT_
.Ltmp163:                               # EH_LABEL
# %bb.106:
.Ltmp164:                               # EH_LABEL
	movq	%rax, %r12
	.cfi_escape 0x2e, 0x00
	movl	$.L.str.14, %esi
	movl	$3, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp165:                               # EH_LABEL
# %bb.107:
	movss	(%r14,%r15,4), %xmm0            # xmm0 = mem[0],zero,zero,zero
	cvtss2sd	%xmm0, %xmm0
.Ltmp166:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%r12, %rdi
	callq	_ZNSo9_M_insertIdEERSoT_
.Ltmp167:                               # EH_LABEL
# %bb.108:
.Ltmp168:                               # EH_LABEL
	movq	%rax, %r12
	.cfi_escape 0x2e, 0x00
	movl	$.L.str.15, %esi
	movl	$12, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp169:                               # EH_LABEL
# %bb.109:
	movss	(%rbx,%r15,4), %xmm0            # xmm0 = mem[0],zero,zero,zero
	mulss	%xmm0, %xmm0
	cvtss2sd	%xmm0, %xmm0
.Ltmp170:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%r12, %rdi
	callq	_ZNSo9_M_insertIdEERSoT_
.Ltmp171:                               # EH_LABEL
# %bb.110:
.Ltmp172:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%rax, %rdi
	movl	$10, %esi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_c
.Ltmp173:                               # EH_LABEL
# %bb.111:
	.cfi_escape 0x2e, 0x00
	movl	$-1, %edi
	callq	exit
.LBB0_122:
	.cfi_escape 0x2e, 0x00
	movl	$_ZSt4cerr, %edi
	movl	$.L.str, %esi
	movl	%eax, %ebx
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	movq	%rax, %r14
	.cfi_escape 0x2e, 0x00
	movl	%ebx, %edi
	callq	hipGetErrorString
	.cfi_escape 0x2e, 0x00
	movq	%r14, %rdi
	movq	%rax, %rsi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	.cfi_escape 0x2e, 0x00
	movl	$.L.str.1, %esi
	movq	%rax, %rdi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	.cfi_escape 0x2e, 0x00
	movl	$.L.str.2, %esi
	movq	%rax, %rdi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	.cfi_escape 0x2e, 0x00
	movq	%rax, %rdi
	movl	$58, %esi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_c
	.cfi_escape 0x2e, 0x00
	movq	%rax, %rdi
	movl	$31, %esi
	callq	_ZNSolsEi
	.cfi_escape 0x2e, 0x00
	movq	%rax, %rdi
	callq	_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_
	.cfi_escape 0x2e, 0x00
	movl	$-1, %edi
	callq	exit
.LBB0_9:
.Ltmp11:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	$_ZSt4cerr, %edi
	movl	$.L.str, %esi
	movl	$31, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp12:                                # EH_LABEL
# %bb.10:
.Ltmp13:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%ebp, %edi
	callq	hipGetErrorString
.Ltmp14:                                # EH_LABEL
# %bb.11:
.Ltmp15:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	$_ZSt4cerr, %edi
	movq	%rax, %rsi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
.Ltmp16:                                # EH_LABEL
# %bb.12:
.Ltmp17:                                # EH_LABEL
	movq	%rax, %r15
	.cfi_escape 0x2e, 0x00
	movl	$.L.str.1, %esi
	movl	$4, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp18:                                # EH_LABEL
# %bb.13:
.Ltmp19:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	$.L.str.2, %esi
	movl	$8, %edx
	movq	%r15, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp20:                                # EH_LABEL
# %bb.14:
.Ltmp21:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%r15, %rdi
	movl	$58, %esi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_c
.Ltmp22:                                # EH_LABEL
# %bb.15:
.Ltmp23:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%rax, %rdi
	movl	$49, %esi
	callq	_ZNSolsEi
.Ltmp24:                                # EH_LABEL
# %bb.16:
.Ltmp25:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%rax, %rdi
	callq	_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_
.Ltmp26:                                # EH_LABEL
# %bb.17:
	.cfi_escape 0x2e, 0x00
	movl	$-1, %edi
	callq	exit
.LBB0_21:
.Ltmp30:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	$_ZSt4cerr, %edi
	movl	$.L.str, %esi
	movl	$31, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp31:                                # EH_LABEL
# %bb.22:
.Ltmp32:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%ebp, %edi
	callq	hipGetErrorString
.Ltmp33:                                # EH_LABEL
# %bb.23:
.Ltmp34:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	$_ZSt4cerr, %edi
	movq	%rax, %rsi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
.Ltmp35:                                # EH_LABEL
# %bb.24:
.Ltmp36:                                # EH_LABEL
	movq	%rax, %r15
	.cfi_escape 0x2e, 0x00
	movl	$.L.str.1, %esi
	movl	$4, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp37:                                # EH_LABEL
# %bb.25:
.Ltmp38:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	$.L.str.2, %esi
	movl	$8, %edx
	movq	%r15, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp39:                                # EH_LABEL
# %bb.26:
.Ltmp40:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%r15, %rdi
	movl	$58, %esi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_c
.Ltmp41:                                # EH_LABEL
# %bb.27:
.Ltmp42:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%rax, %rdi
	movl	$50, %esi
	callq	_ZNSolsEi
.Ltmp43:                                # EH_LABEL
# %bb.28:
.Ltmp44:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%rax, %rdi
	callq	_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_
.Ltmp45:                                # EH_LABEL
# %bb.29:
	.cfi_escape 0x2e, 0x00
	movl	$-1, %edi
	callq	exit
.LBB0_34:
.Ltmp52:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	$_ZSt4cerr, %edi
	movl	$.L.str, %esi
	movl	$31, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp53:                                # EH_LABEL
# %bb.35:
.Ltmp54:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%ebp, %edi
	callq	hipGetErrorString
.Ltmp55:                                # EH_LABEL
# %bb.36:
.Ltmp56:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	$_ZSt4cerr, %edi
	movq	%rax, %rsi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
.Ltmp57:                                # EH_LABEL
# %bb.37:
.Ltmp58:                                # EH_LABEL
	movq	%rax, %r15
	.cfi_escape 0x2e, 0x00
	movl	$.L.str.1, %esi
	movl	$4, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp59:                                # EH_LABEL
# %bb.38:
.Ltmp60:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	$.L.str.2, %esi
	movl	$8, %edx
	movq	%r15, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp61:                                # EH_LABEL
# %bb.39:
.Ltmp62:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%r15, %rdi
	movl	$58, %esi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_c
.Ltmp63:                                # EH_LABEL
# %bb.40:
.Ltmp64:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%rax, %rdi
	movl	$55, %esi
	callq	_ZNSolsEi
.Ltmp65:                                # EH_LABEL
# %bb.41:
.Ltmp66:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%rax, %rdi
	callq	_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_
.Ltmp67:                                # EH_LABEL
# %bb.42:
	.cfi_escape 0x2e, 0x00
	movl	$-1, %edi
	callq	exit
.LBB0_51:
.Ltmp79:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	$_ZSt4cerr, %edi
	movl	$.L.str, %esi
	movl	$31, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp80:                                # EH_LABEL
# %bb.52:
.Ltmp81:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%ebp, %edi
	callq	hipGetErrorString
.Ltmp82:                                # EH_LABEL
# %bb.53:
.Ltmp83:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	$_ZSt4cerr, %edi
	movq	%rax, %rsi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
.Ltmp84:                                # EH_LABEL
# %bb.54:
.Ltmp85:                                # EH_LABEL
	movq	%rax, %r15
	.cfi_escape 0x2e, 0x00
	movl	$.L.str.1, %esi
	movl	$4, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp86:                                # EH_LABEL
# %bb.55:
.Ltmp87:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	$.L.str.2, %esi
	movl	$8, %edx
	movq	%r15, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp88:                                # EH_LABEL
# %bb.56:
.Ltmp89:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%r15, %rdi
	movl	$58, %esi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_c
.Ltmp90:                                # EH_LABEL
# %bb.57:
.Ltmp91:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%rax, %rdi
	movl	$66, %esi
	callq	_ZNSolsEi
.Ltmp92:                                # EH_LABEL
# %bb.58:
.Ltmp93:                                # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%rax, %rdi
	callq	_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_
.Ltmp94:                                # EH_LABEL
# %bb.59:
	.cfi_escape 0x2e, 0x00
	movl	$-1, %edi
	callq	exit
.LBB0_64:
.Ltmp100:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	$_ZSt4cerr, %edi
	movl	$.L.str, %esi
	movl	$31, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp101:                               # EH_LABEL
# %bb.65:
.Ltmp102:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%ebp, %edi
	callq	hipGetErrorString
.Ltmp103:                               # EH_LABEL
# %bb.66:
.Ltmp104:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	$_ZSt4cerr, %edi
	movq	%rax, %rsi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
.Ltmp105:                               # EH_LABEL
# %bb.67:
.Ltmp106:                               # EH_LABEL
	movq	%rax, %r15
	.cfi_escape 0x2e, 0x00
	movl	$.L.str.1, %esi
	movl	$4, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp107:                               # EH_LABEL
# %bb.68:
.Ltmp108:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	$.L.str.2, %esi
	movl	$8, %edx
	movq	%r15, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp109:                               # EH_LABEL
# %bb.69:
.Ltmp110:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%r15, %rdi
	movl	$58, %esi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_c
.Ltmp111:                               # EH_LABEL
# %bb.70:
.Ltmp112:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%rax, %rdi
	movl	$69, %esi
	callq	_ZNSolsEi
.Ltmp113:                               # EH_LABEL
# %bb.71:
.Ltmp114:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%rax, %rdi
	callq	_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_
.Ltmp115:                               # EH_LABEL
# %bb.72:
	.cfi_escape 0x2e, 0x00
	movl	$-1, %edi
	callq	exit
.LBB0_76:
.Ltmp119:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	$_ZSt4cerr, %edi
	movl	$.L.str, %esi
	movl	$31, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp120:                               # EH_LABEL
# %bb.77:
.Ltmp121:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%ebp, %edi
	callq	hipGetErrorString
.Ltmp122:                               # EH_LABEL
# %bb.78:
.Ltmp123:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	$_ZSt4cerr, %edi
	movq	%rax, %rsi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
.Ltmp124:                               # EH_LABEL
# %bb.79:
.Ltmp125:                               # EH_LABEL
	movq	%rax, %r15
	.cfi_escape 0x2e, 0x00
	movl	$.L.str.1, %esi
	movl	$4, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp126:                               # EH_LABEL
# %bb.80:
.Ltmp127:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	$.L.str.2, %esi
	movl	$8, %edx
	movq	%r15, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp128:                               # EH_LABEL
# %bb.81:
.Ltmp129:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%r15, %rdi
	movl	$58, %esi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_c
.Ltmp130:                               # EH_LABEL
# %bb.82:
.Ltmp131:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%rax, %rdi
	movl	$71, %esi
	callq	_ZNSolsEi
.Ltmp132:                               # EH_LABEL
# %bb.83:
.Ltmp133:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%rax, %rdi
	callq	_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_
.Ltmp134:                               # EH_LABEL
# %bb.84:
	.cfi_escape 0x2e, 0x00
	movl	$-1, %edi
	callq	exit
.LBB0_88:
.Ltmp138:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	$_ZSt4cerr, %edi
	movl	$.L.str, %esi
	movl	$31, %edx
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp139:                               # EH_LABEL
# %bb.89:
.Ltmp140:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	%ebp, %edi
	callq	hipGetErrorString
.Ltmp141:                               # EH_LABEL
# %bb.90:
.Ltmp142:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	$_ZSt4cerr, %edi
	movq	%rax, %rsi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
.Ltmp143:                               # EH_LABEL
# %bb.91:
.Ltmp144:                               # EH_LABEL
	movq	%rax, %r15
	.cfi_escape 0x2e, 0x00
	movl	$.L.str.1, %esi
	movl	$4, %edx
	movq	%rax, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp145:                               # EH_LABEL
# %bb.92:
.Ltmp146:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movl	$.L.str.2, %esi
	movl	$8, %edx
	movq	%r15, %rdi
	callq	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
.Ltmp147:                               # EH_LABEL
# %bb.93:
.Ltmp148:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%r15, %rdi
	movl	$58, %esi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_c
.Ltmp149:                               # EH_LABEL
# %bb.94:
.Ltmp150:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%rax, %rdi
	movl	$72, %esi
	callq	_ZNSolsEi
.Ltmp151:                               # EH_LABEL
# %bb.95:
.Ltmp152:                               # EH_LABEL
	.cfi_escape 0x2e, 0x00
	movq	%rax, %rdi
	callq	_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_
.Ltmp153:                               # EH_LABEL
# %bb.96:
	.cfi_escape 0x2e, 0x00
	movl	$-1, %edi
	callq	exit
.LBB0_123:
.Ltmp2:                                 # EH_LABEL
	movq	%rax, %r15
	jmp	.LBB0_121
.LBB0_97:
.Ltmp154:                               # EH_LABEL
	jmp	.LBB0_120
.LBB0_85:
.Ltmp135:                               # EH_LABEL
	jmp	.LBB0_120
.LBB0_73:
.Ltmp116:                               # EH_LABEL
	jmp	.LBB0_120
.LBB0_60:
.Ltmp95:                                # EH_LABEL
	jmp	.LBB0_120
.LBB0_43:
.Ltmp68:                                # EH_LABEL
	jmp	.LBB0_120
.LBB0_30:
.Ltmp46:                                # EH_LABEL
	jmp	.LBB0_120
.LBB0_18:
.Ltmp27:                                # EH_LABEL
	jmp	.LBB0_120
.LBB0_119:
.Ltmp49:                                # EH_LABEL
	jmp	.LBB0_120
.LBB0_124:
.Ltmp159:                               # EH_LABEL
	jmp	.LBB0_120
.LBB0_112:
.Ltmp174:                               # EH_LABEL
.LBB0_120:
	movq	%rax, %r15
	.cfi_escape 0x2e, 0x00
	movl	$4000000, %esi                  # imm = 0x3D0900
	movq	%r14, %rdi
	callq	_ZdlPvm
.LBB0_121:
	.cfi_escape 0x2e, 0x00
	movl	$4000000, %esi                  # imm = 0x3D0900
	movq	%rbx, %rdi
	callq	_ZdlPvm
	.cfi_escape 0x2e, 0x00
	movq	%r15, %rdi
	callq	_Unwind_Resume@PLT
.Lfunc_end0:
	.size	main, .Lfunc_end0-main
	.cfi_endproc
	.section	.gcc_except_table,"a",@progbits
	.p2align	2, 0x0
GCC_except_table0:
.Lexception0:
	.byte	255                             # @LPStart Encoding = omit
	.byte	255                             # @TType Encoding = omit
	.byte	1                               # Call site Encoding = uleb128
	.uleb128 .Lcst_end0-.Lcst_begin0
.Lcst_begin0:
	.uleb128 .Lfunc_begin0-.Lfunc_begin0    # >> Call Site 1 <<
	.uleb128 .Ltmp0-.Lfunc_begin0           #   Call between .Lfunc_begin0 and .Ltmp0
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp0-.Lfunc_begin0           # >> Call Site 2 <<
	.uleb128 .Ltmp1-.Ltmp0                  #   Call between .Ltmp0 and .Ltmp1
	.uleb128 .Ltmp2-.Lfunc_begin0           #     jumps to .Ltmp2
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp1-.Lfunc_begin0           # >> Call Site 3 <<
	.uleb128 .Ltmp3-.Ltmp1                  #   Call between .Ltmp1 and .Ltmp3
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp3-.Lfunc_begin0           # >> Call Site 4 <<
	.uleb128 .Ltmp8-.Ltmp3                  #   Call between .Ltmp3 and .Ltmp8
	.uleb128 .Ltmp49-.Lfunc_begin0          #     jumps to .Ltmp49
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp9-.Lfunc_begin0           # >> Call Site 5 <<
	.uleb128 .Ltmp10-.Ltmp9                 #   Call between .Ltmp9 and .Ltmp10
	.uleb128 .Ltmp27-.Lfunc_begin0          #     jumps to .Ltmp27
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp28-.Lfunc_begin0          # >> Call Site 6 <<
	.uleb128 .Ltmp29-.Ltmp28                #   Call between .Ltmp28 and .Ltmp29
	.uleb128 .Ltmp46-.Lfunc_begin0          #     jumps to .Ltmp46
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp47-.Lfunc_begin0          # >> Call Site 7 <<
	.uleb128 .Ltmp48-.Ltmp47                #   Call between .Ltmp47 and .Ltmp48
	.uleb128 .Ltmp49-.Lfunc_begin0          #     jumps to .Ltmp49
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp50-.Lfunc_begin0          # >> Call Site 8 <<
	.uleb128 .Ltmp51-.Ltmp50                #   Call between .Ltmp50 and .Ltmp51
	.uleb128 .Ltmp68-.Lfunc_begin0          #     jumps to .Ltmp68
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp69-.Lfunc_begin0          # >> Call Site 9 <<
	.uleb128 .Ltmp76-.Ltmp69                #   Call between .Ltmp69 and .Ltmp76
	.uleb128 .Ltmp159-.Lfunc_begin0         #     jumps to .Ltmp159
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp77-.Lfunc_begin0          # >> Call Site 10 <<
	.uleb128 .Ltmp78-.Ltmp77                #   Call between .Ltmp77 and .Ltmp78
	.uleb128 .Ltmp95-.Lfunc_begin0          #     jumps to .Ltmp95
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp96-.Lfunc_begin0          # >> Call Site 11 <<
	.uleb128 .Ltmp97-.Ltmp96                #   Call between .Ltmp96 and .Ltmp97
	.uleb128 .Ltmp159-.Lfunc_begin0         #     jumps to .Ltmp159
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp98-.Lfunc_begin0          # >> Call Site 12 <<
	.uleb128 .Ltmp99-.Ltmp98                #   Call between .Ltmp98 and .Ltmp99
	.uleb128 .Ltmp116-.Lfunc_begin0         #     jumps to .Ltmp116
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp117-.Lfunc_begin0         # >> Call Site 13 <<
	.uleb128 .Ltmp118-.Ltmp117              #   Call between .Ltmp117 and .Ltmp118
	.uleb128 .Ltmp135-.Lfunc_begin0         #     jumps to .Ltmp135
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp136-.Lfunc_begin0         # >> Call Site 14 <<
	.uleb128 .Ltmp137-.Ltmp136              #   Call between .Ltmp136 and .Ltmp137
	.uleb128 .Ltmp154-.Lfunc_begin0         #     jumps to .Ltmp154
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp155-.Lfunc_begin0         # >> Call Site 15 <<
	.uleb128 .Ltmp158-.Ltmp155              #   Call between .Ltmp155 and .Ltmp158
	.uleb128 .Ltmp159-.Lfunc_begin0         #     jumps to .Ltmp159
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp160-.Lfunc_begin0         # >> Call Site 16 <<
	.uleb128 .Ltmp173-.Ltmp160              #   Call between .Ltmp160 and .Ltmp173
	.uleb128 .Ltmp174-.Lfunc_begin0         #     jumps to .Ltmp174
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp173-.Lfunc_begin0         # >> Call Site 17 <<
	.uleb128 .Ltmp11-.Ltmp173               #   Call between .Ltmp173 and .Ltmp11
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp11-.Lfunc_begin0          # >> Call Site 18 <<
	.uleb128 .Ltmp26-.Ltmp11                #   Call between .Ltmp11 and .Ltmp26
	.uleb128 .Ltmp27-.Lfunc_begin0          #     jumps to .Ltmp27
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp30-.Lfunc_begin0          # >> Call Site 19 <<
	.uleb128 .Ltmp45-.Ltmp30                #   Call between .Ltmp30 and .Ltmp45
	.uleb128 .Ltmp46-.Lfunc_begin0          #     jumps to .Ltmp46
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp52-.Lfunc_begin0          # >> Call Site 20 <<
	.uleb128 .Ltmp67-.Ltmp52                #   Call between .Ltmp52 and .Ltmp67
	.uleb128 .Ltmp68-.Lfunc_begin0          #     jumps to .Ltmp68
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp79-.Lfunc_begin0          # >> Call Site 21 <<
	.uleb128 .Ltmp94-.Ltmp79                #   Call between .Ltmp79 and .Ltmp94
	.uleb128 .Ltmp95-.Lfunc_begin0          #     jumps to .Ltmp95
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp100-.Lfunc_begin0         # >> Call Site 22 <<
	.uleb128 .Ltmp115-.Ltmp100              #   Call between .Ltmp100 and .Ltmp115
	.uleb128 .Ltmp116-.Lfunc_begin0         #     jumps to .Ltmp116
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp119-.Lfunc_begin0         # >> Call Site 23 <<
	.uleb128 .Ltmp134-.Ltmp119              #   Call between .Ltmp119 and .Ltmp134
	.uleb128 .Ltmp135-.Lfunc_begin0         #     jumps to .Ltmp135
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp138-.Lfunc_begin0         # >> Call Site 24 <<
	.uleb128 .Ltmp153-.Ltmp138              #   Call between .Ltmp138 and .Ltmp153
	.uleb128 .Ltmp154-.Lfunc_begin0         #     jumps to .Ltmp154
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp153-.Lfunc_begin0         # >> Call Site 25 <<
	.uleb128 .Lfunc_end0-.Ltmp153           #   Call between .Ltmp153 and .Lfunc_end0
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
.Lcst_end0:
	.p2align	2, 0x0
                                        # -- End function
	.section	.text._Z35__device_stub__vector_square_kernelIfEvPT_S1_y,"axG",@progbits,_Z35__device_stub__vector_square_kernelIfEvPT_S1_y,comdat
	.weak	_Z35__device_stub__vector_square_kernelIfEvPT_S1_y # -- Begin function _Z35__device_stub__vector_square_kernelIfEvPT_S1_y
	.p2align	4
	.type	_Z35__device_stub__vector_square_kernelIfEvPT_S1_y,@function
_Z35__device_stub__vector_square_kernelIfEvPT_S1_y: # @_Z35__device_stub__vector_square_kernelIfEvPT_S1_y
	.cfi_startproc
# %bb.0:
	subq	$104, %rsp
	.cfi_def_cfa_offset 112
	movq	%rdi, 72(%rsp)
	movq	%rsi, 64(%rsp)
	movq	%rdx, 56(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 80(%rsp)
	leaq	64(%rsp), %rax
	movq	%rax, 88(%rsp)
	leaq	56(%rsp), %rax
	movq	%rax, 96(%rsp)
	leaq	40(%rsp), %rdi
	leaq	24(%rsp), %rsi
	leaq	16(%rsp), %rdx
	leaq	8(%rsp), %rcx
	callq	__hipPopCallConfiguration
	movq	40(%rsp), %rsi
	movl	48(%rsp), %edx
	movq	24(%rsp), %rcx
	movl	32(%rsp), %r8d
	leaq	80(%rsp), %r9
	movl	$_Z20vector_square_kernelIfEvPT_S1_y, %edi
	pushq	8(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	24(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel
	addq	$120, %rsp
	.cfi_adjust_cfa_offset -120
	retq
.Lfunc_end1:
	.size	_Z35__device_stub__vector_square_kernelIfEvPT_S1_y, .Lfunc_end1-_Z35__device_stub__vector_square_kernelIfEvPT_S1_y
	.cfi_endproc
                                        # -- End function
	.text
	.p2align	4                               # -- Begin function __hip_module_ctor
	.type	__hip_module_ctor,@function
__hip_module_ctor:                      # @__hip_module_ctor
	.cfi_startproc
# %bb.0:
	subq	$40, %rsp
	.cfi_def_cfa_offset 48
	movq	__hip_gpubin_handle_9355bdbd540905a(%rip), %rdi
	testq	%rdi, %rdi
	jne	.LBB2_2
# %bb.1:
	movl	$__hip_fatbin_wrapper, %edi
	callq	__hipRegisterFatBinary
	movq	%rax, %rdi
	movq	%rax, __hip_gpubin_handle_9355bdbd540905a(%rip)
.LBB2_2:
	xorps	%xmm0, %xmm0
	movups	%xmm0, 16(%rsp)
	movups	%xmm0, (%rsp)
	movl	$_Z20vector_square_kernelIfEvPT_S1_y, %esi
	movl	$.L__unnamed_1, %edx
	movl	$.L__unnamed_1, %ecx
	movl	$-1, %r8d
	xorl	%r9d, %r9d
	callq	__hipRegisterFunction
	movl	$__hip_module_dtor, %edi
	addq	$40, %rsp
	.cfi_def_cfa_offset 8
	jmp	atexit                          # TAILCALL
.Lfunc_end2:
	.size	__hip_module_ctor, .Lfunc_end2-__hip_module_ctor
	.cfi_endproc
                                        # -- End function
	.p2align	4                               # -- Begin function __hip_module_dtor
	.type	__hip_module_dtor,@function
__hip_module_dtor:                      # @__hip_module_dtor
	.cfi_startproc
# %bb.0:
	movq	__hip_gpubin_handle_9355bdbd540905a(%rip), %rdi
	testq	%rdi, %rdi
	je	.LBB3_2
# %bb.1:
	pushq	%rax
	.cfi_def_cfa_offset 16
	callq	__hipUnregisterFatBinary
	movq	$0, __hip_gpubin_handle_9355bdbd540905a(%rip)
	addq	$8, %rsp
	.cfi_def_cfa_offset 8
.LBB3_2:
	retq
.Lfunc_end3:
	.size	__hip_module_dtor, .Lfunc_end3-__hip_module_dtor
	.cfi_endproc
                                        # -- End function
	.type	.L.str,@object                  # @.str
	.section	.rodata.str1.1,"aMS",@progbits,1
.L.str:
	.asciz	"An error occured encountered: \""
	.size	.L.str, 32

	.type	.L.str.1,@object                # @.str.1
.L.str.1:
	.asciz	"\" at"
	.size	.L.str.1, 5

	.type	.L.str.2,@object                # @.str.2
.L.str.2:
	.asciz	"main.hip"
	.size	.L.str.2, 9

	.type	.L.str.3,@object                # @.str.3
.L.str.3:
	.asciz	"Info: running on device"
	.size	.L.str.3, 24

	.type	.L.str.4,@object                # @.str.4
.L.str.4:
	.asciz	"\n"
	.size	.L.str.4, 2

	.type	.L.str.5,@object                # @.str.5
.L.str.5:
	.asciz	"Info: allocate host mem ("
	.size	.L.str.5, 26

	.type	.L.str.6,@object                # @.str.6
.L.str.6:
	.asciz	"MiB) "
	.size	.L.str.6, 6

	.type	.L.str.7,@object                # @.str.7
.L.str.7:
	.asciz	"info: allocate device mem ("
	.size	.L.str.7, 28

	.type	.L.str.8,@object                # @.str.8
.L.str.8:
	.asciz	"MiB)\n"
	.size	.L.str.8, 6

	.type	.L.str.9,@object                # @.str.9
.L.str.9:
	.asciz	"info: copy Host2Device\n"
	.size	.L.str.9, 24

	.type	.L.str.10,@object               # @.str.10
.L.str.10:
	.asciz	"info: launch 'vector_square_kernel' kernel\n"
	.size	.L.str.10, 44

	.type	_Z20vector_square_kernelIfEvPT_S1_y,@object # @_Z20vector_square_kernelIfEvPT_S1_y
	.section	.rodata._Z20vector_square_kernelIfEvPT_S1_y,"aG",@progbits,_Z20vector_square_kernelIfEvPT_S1_y,comdat
	.weak	_Z20vector_square_kernelIfEvPT_S1_y
	.p2align	3, 0x0
_Z20vector_square_kernelIfEvPT_S1_y:
	.quad	_Z35__device_stub__vector_square_kernelIfEvPT_S1_y
	.size	_Z20vector_square_kernelIfEvPT_S1_y, 8

	.type	.L.str.11,@object               # @.str.11
	.section	.rodata.str1.1,"aMS",@progbits,1
.L.str.11:
	.asciz	"info: copy Device2Host\n"
	.size	.L.str.11, 24

	.type	.L.str.12,@object               # @.str.12
.L.str.12:
	.asciz	"info: check result\n"
	.size	.L.str.12, 20

	.type	.L.str.13,@object               # @.str.13
.L.str.13:
	.asciz	"FAILED! h_out ["
	.size	.L.str.13, 16

	.type	.L.str.14,@object               # @.str.14
.L.str.14:
	.asciz	"] ="
	.size	.L.str.14, 4

	.type	.L.str.15,@object               # @.str.15
.L.str.15:
	.asciz	", expected: "
	.size	.L.str.15, 13

	.type	.L.str.16,@object               # @.str.16
.L.str.16:
	.asciz	"PASSED!\n"
	.size	.L.str.16, 9

	.type	.L__unnamed_1,@object           # @0
.L__unnamed_1:
	.asciz	"_Z20vector_square_kernelIfEvPT_S1_y"
	.size	.L__unnamed_1, 36

	.type	__hip_fatbin_wrapper,@object    # @__hip_fatbin_wrapper
	.section	.hipFatBinSegment,"a",@progbits
	.p2align	3, 0x0
__hip_fatbin_wrapper:
	.long	1212764230                      # 0x48495046
	.long	1                               # 0x1
	.quad	__hip_fatbin_9355bdbd540905a
	.quad	0
	.size	__hip_fatbin_wrapper, 24

	.type	__hip_gpubin_handle_9355bdbd540905a,@object # @__hip_gpubin_handle_9355bdbd540905a
	.local	__hip_gpubin_handle_9355bdbd540905a
	.comm	__hip_gpubin_handle_9355bdbd540905a,8,8
	.section	.init_array,"aw",@init_array
	.p2align	3, 0x0
	.quad	__hip_module_ctor
	.type	__hip_cuid_9355bdbd540905a,@object # @__hip_cuid_9355bdbd540905a
	.bss
	.globl	__hip_cuid_9355bdbd540905a
__hip_cuid_9355bdbd540905a:
	.byte	0                               # 0x0
	.size	__hip_cuid_9355bdbd540905a, 1

	.ident	"AMD clang version 22.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.2.0 26014 7b800a19466229b8479a78de19143dc33c3ab9b5)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __gxx_personality_v0
	.addrsig_sym _Z35__device_stub__vector_square_kernelIfEvPT_S1_y
	.addrsig_sym __hip_module_ctor
	.addrsig_sym __hip_module_dtor
	.addrsig_sym _Unwind_Resume
	.addrsig_sym _ZSt4cerr
	.addrsig_sym _ZSt4cout
	.addrsig_sym _Z20vector_square_kernelIfEvPT_S1_y
	.addrsig_sym __hip_fatbin_9355bdbd540905a
	.addrsig_sym __hip_fatbin_wrapper
	.addrsig_sym __hip_cuid_9355bdbd540905a

# __CLANG_OFFLOAD_BUNDLE____END__ host-x86_64-unknown-linux-gnu-
