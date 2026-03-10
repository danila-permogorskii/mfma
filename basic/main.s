
# __CLANG_OFFLOAD_BUNDLE____START__ hip-amdgcn-amd-amdhsa--gfx942
	.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
	.amdhsa_code_object_version 6
	.text
	.weak	__cxa_pure_virtual              ; -- Begin function __cxa_pure_virtual
	.p2align	2
	.type	__cxa_pure_virtual,@function
__cxa_pure_virtual:                     ; @__cxa_pure_virtual
; %bb.0:
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
	s_mov_b32 s0, s33
	s_mov_b32 s33, s32
	s_trap 2
.Lfunc_end0:
	.size	__cxa_pure_virtual, .Lfunc_end0-__cxa_pure_virtual
                                        ; -- End function
	.set __cxa_pure_virtual.num_vgpr, 0
	.set __cxa_pure_virtual.num_agpr, 0
	.set __cxa_pure_virtual.numbered_sgpr, 34
	.set __cxa_pure_virtual.num_named_barrier, 0
	.set __cxa_pure_virtual.private_seg_size, 0
	.set __cxa_pure_virtual.uses_vcc, 0
	.set __cxa_pure_virtual.uses_flat_scratch, 0
	.set __cxa_pure_virtual.has_dyn_sized_stack, 0
	.set __cxa_pure_virtual.has_recursion, 0
	.set __cxa_pure_virtual.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Function info:
; codeLenInByte = 16
; TotalNumSgprs: 40
; NumVgprs: 0
; NumAgprs: 0
; TotalNumVgprs: 0
; ScratchSize: 0
; MemoryBound: 0
	.text
	.weak	__cxa_deleted_virtual           ; -- Begin function __cxa_deleted_virtual
	.p2align	2
	.type	__cxa_deleted_virtual,@function
__cxa_deleted_virtual:                  ; @__cxa_deleted_virtual
; %bb.0:
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
	s_mov_b32 s0, s33
	s_mov_b32 s33, s32
	s_trap 2
.Lfunc_end1:
	.size	__cxa_deleted_virtual, .Lfunc_end1-__cxa_deleted_virtual
                                        ; -- End function
	.set __cxa_deleted_virtual.num_vgpr, 0
	.set __cxa_deleted_virtual.num_agpr, 0
	.set __cxa_deleted_virtual.numbered_sgpr, 34
	.set __cxa_deleted_virtual.num_named_barrier, 0
	.set __cxa_deleted_virtual.private_seg_size, 0
	.set __cxa_deleted_virtual.uses_vcc, 0
	.set __cxa_deleted_virtual.uses_flat_scratch, 0
	.set __cxa_deleted_virtual.has_dyn_sized_stack, 0
	.set __cxa_deleted_virtual.has_recursion, 0
	.set __cxa_deleted_virtual.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Function info:
; codeLenInByte = 16
; TotalNumSgprs: 40
; NumVgprs: 0
; NumAgprs: 0
; TotalNumVgprs: 0
; ScratchSize: 0
; MemoryBound: 0
	.text
	.p2align	2                               ; -- Begin function __ockl_hsa_signal_add
	.type	__ockl_hsa_signal_add,@function
__ockl_hsa_signal_add:                  ; @__ockl_hsa_signal_add
; %bb.0:
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
	s_mov_b32 s12, s33
	s_mov_b32 s33, s32
	s_xor_saveexec_b64 s[0:1], -1
	scratch_store_dword off, v6, s33 offset:44 ; 4-byte Folded Spill
	s_mov_b64 exec, s[0:1]
	s_add_i32 s32, s32, 52
	scratch_store_dword off, v4, s33 offset:32 ; 4-byte Folded Spill
	scratch_store_dword off, v3, s33 offset:28 ; 4-byte Folded Spill
	v_mov_b32_e32 v4, v1
	scratch_load_dword v1, off, s33 offset:28 ; 4-byte Folded Reload
                                        ; kill: def $vgpr2 killed $vgpr2 def $vgpr2_vgpr3 killed $exec
	s_waitcnt vmcnt(0)
	v_mov_b32_e32 v3, v1
                                        ; kill: def $vgpr0 killed $vgpr0 def $vgpr0_vgpr1 killed $exec
	v_mov_b32_e32 v1, v4
	scratch_store_dwordx2 off, v[2:3], s33 offset:20 ; 8-byte Folded Spill
	v_mov_b64_e32 v[2:3], v[0:1]
	scratch_store_dwordx2 off, v[2:3], s33 offset:12 ; 8-byte Folded Spill
	s_mov_b64 s[0:1], 8
	v_lshl_add_u64 v[0:1], v[0:1], 0, s[0:1]
	scratch_store_dwordx2 off, v[0:1], s33 offset:4 ; 8-byte Folded Spill
; %bb.1:
	scratch_load_dword v0, off, s33 offset:32 ; 4-byte Folded Reload
	s_mov_b32 s0, 3
	s_waitcnt vmcnt(0)
	v_cmp_gt_i32_e64 s[0:1], v0, s0
	s_mov_b64 s[2:3], 0
                                        ; implicit-def: $vgpr6 : SGPR spill to VGPR lane
	v_writelane_b32 v6, s2, 0
	s_nop 1
	v_writelane_b32 v6, s3, 1
	s_mov_b64 s[2:3], exec
	s_and_b64 s[0:1], s[2:3], s[0:1]
	s_xor_b64 s[2:3], s[0:1], s[2:3]
	v_writelane_b32 v6, s2, 2
	s_nop 1
	v_writelane_b32 v6, s3, 3
	s_or_saveexec_b64 s[10:11], -1
	scratch_store_dword off, v6, s33        ; 4-byte Folded Spill
	s_mov_b64 exec, s[10:11]
	s_mov_b64 exec, s[0:1]
	s_cbranch_execz .LBB2_3
; %bb.2:
	s_or_saveexec_b64 s[10:11], -1
	scratch_load_dword v6, off, s33         ; 4-byte Folded Reload
	s_mov_b64 exec, s[10:11]
	scratch_load_dword v0, off, s33 offset:32 ; 4-byte Folded Reload
	s_mov_b32 s0, 4
	s_waitcnt vmcnt(0)
	v_cmp_gt_i32_e64 s[0:1], v0, s0
	s_mov_b64 s[2:3], 0
	v_writelane_b32 v6, s2, 4
	s_nop 1
	v_writelane_b32 v6, s3, 5
	s_mov_b64 s[2:3], exec
	s_and_b64 s[0:1], s[2:3], s[0:1]
	s_xor_b64 s[2:3], s[0:1], s[2:3]
	v_writelane_b32 v6, s2, 6
	s_nop 1
	v_writelane_b32 v6, s3, 7
	s_or_saveexec_b64 s[10:11], -1
	scratch_store_dword off, v6, s33        ; 4-byte Folded Spill
	s_mov_b64 exec, s[10:11]
	s_mov_b64 exec, s[0:1]
	s_cbranch_execz .LBB2_17
	s_branch .LBB2_4
.LBB2_3:
	s_or_saveexec_b64 s[10:11], -1
	scratch_load_dword v6, off, s33         ; 4-byte Folded Reload
	s_mov_b64 exec, s[10:11]
	s_waitcnt vmcnt(0)
	v_readlane_b32 s0, v6, 2
	v_readlane_b32 s1, v6, 3
	s_or_saveexec_b64 s[0:1], s[0:1]
	v_readlane_b32 s4, v6, 0
	v_readlane_b32 s5, v6, 1
	s_nop 0
	v_writelane_b32 v6, s4, 8
	s_nop 1
	v_writelane_b32 v6, s5, 9
	s_mov_b64 s[2:3], 0
	v_writelane_b32 v6, s4, 10
	s_nop 1
	v_writelane_b32 v6, s5, 11
	v_writelane_b32 v6, s2, 12
	s_nop 1
	v_writelane_b32 v6, s3, 13
	s_and_b64 s[0:1], exec, s[0:1]
	v_writelane_b32 v6, s0, 14
	s_nop 1
	v_writelane_b32 v6, s1, 15
	s_or_saveexec_b64 s[10:11], -1
	scratch_store_dword off, v6, s33        ; 4-byte Folded Spill
	s_mov_b64 exec, s[10:11]
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execz .LBB2_13
	s_branch .LBB2_6
.LBB2_4:
	s_or_saveexec_b64 s[10:11], -1
	scratch_load_dword v6, off, s33         ; 4-byte Folded Reload
	s_mov_b64 exec, s[10:11]
	scratch_load_dword v0, off, s33 offset:32 ; 4-byte Folded Reload
	s_mov_b32 s0, 5
	s_waitcnt vmcnt(0)
	v_cmp_eq_u32_e64 s[2:3], v0, s0
	s_mov_b64 s[0:1], -1
	v_writelane_b32 v6, s0, 16
	s_nop 1
	v_writelane_b32 v6, s1, 17
	s_mov_b64 s[0:1], exec
	v_writelane_b32 v6, s0, 18
	s_nop 1
	v_writelane_b32 v6, s1, 19
	s_or_saveexec_b64 s[10:11], -1
	scratch_store_dword off, v6, s33        ; 4-byte Folded Spill
	s_mov_b64 exec, s[10:11]
	s_and_b64 s[0:1], s[0:1], s[2:3]
	s_mov_b64 exec, s[0:1]
	s_cbranch_execz .LBB2_15
	s_branch .LBB2_18
.LBB2_5:
	s_or_saveexec_b64 s[10:11], -1
	scratch_load_dword v6, off, s33         ; 4-byte Folded Reload
	s_mov_b64 exec, s[10:11]
	s_waitcnt vmcnt(0)
	v_readlane_b32 s2, v6, 20
	v_readlane_b32 s3, v6, 21
	s_or_b64 exec, exec, s[2:3]
	v_readlane_b32 s0, v6, 22
	v_readlane_b32 s1, v6, 23
	s_and_b64 s[0:1], s[0:1], exec
	v_writelane_b32 v6, s0, 0
	s_nop 1
	v_writelane_b32 v6, s1, 1
	s_or_saveexec_b64 s[10:11], -1
	scratch_store_dword off, v6, s33        ; 4-byte Folded Spill
	s_mov_b64 exec, s[10:11]
	s_branch .LBB2_3
.LBB2_6:
	s_or_saveexec_b64 s[10:11], -1
	scratch_load_dword v6, off, s33         ; 4-byte Folded Reload
	s_mov_b64 exec, s[10:11]
	scratch_load_dword v0, off, s33 offset:32 ; 4-byte Folded Reload
	s_mov_b32 s0, 2
	s_waitcnt vmcnt(0)
	v_cmp_gt_i32_e64 s[0:1], v0, s0
	s_mov_b64 s[2:3], exec
	s_and_b64 s[0:1], s[2:3], s[0:1]
	s_xor_b64 s[2:3], s[0:1], s[2:3]
	v_writelane_b32 v6, s2, 24
	s_nop 1
	v_writelane_b32 v6, s3, 25
	s_or_saveexec_b64 s[10:11], -1
	scratch_store_dword off, v6, s33        ; 4-byte Folded Spill
	s_mov_b64 exec, s[10:11]
	s_mov_b64 exec, s[0:1]
	s_cbranch_execz .LBB2_7
	s_branch .LBB2_14
.LBB2_7:
	s_or_saveexec_b64 s[10:11], -1
	scratch_load_dword v6, off, s33         ; 4-byte Folded Reload
	s_mov_b64 exec, s[10:11]
	s_waitcnt vmcnt(0)
	v_readlane_b32 s0, v6, 24
	v_readlane_b32 s1, v6, 25
	s_or_saveexec_b64 s[0:1], s[0:1]
	v_readlane_b32 s4, v6, 8
	v_readlane_b32 s5, v6, 9
	s_mov_b64 s[2:3], 0
	v_writelane_b32 v6, s4, 26
	s_nop 1
	v_writelane_b32 v6, s5, 27
	v_writelane_b32 v6, s2, 28
	s_nop 1
	v_writelane_b32 v6, s3, 29
	s_and_b64 s[0:1], exec, s[0:1]
	v_writelane_b32 v6, s0, 30
	s_nop 1
	v_writelane_b32 v6, s1, 31
	s_or_saveexec_b64 s[10:11], -1
	scratch_store_dword off, v6, s33        ; 4-byte Folded Spill
	s_mov_b64 exec, s[10:11]
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execz .LBB2_9
; %bb.8:
	s_or_saveexec_b64 s[10:11], -1
	scratch_load_dword v6, off, s33         ; 4-byte Folded Reload
	s_mov_b64 exec, s[10:11]
	s_waitcnt vmcnt(0)
	v_readlane_b32 s2, v6, 8
	v_readlane_b32 s3, v6, 9
	scratch_load_dword v0, off, s33 offset:32 ; 4-byte Folded Reload
	s_mov_b32 s0, 1
	s_waitcnt vmcnt(0)
	v_cmp_lt_i32_e64 s[4:5], v0, s0
	s_mov_b64 s[0:1], -1
	s_mov_b64 s[0:1], exec
	s_andn2_b64 s[2:3], s[2:3], exec
	s_and_b64 s[4:5], s[4:5], exec
	s_or_b64 s[2:3], s[2:3], s[4:5]
	v_writelane_b32 v6, s2, 26
	s_nop 1
	v_writelane_b32 v6, s3, 27
	v_writelane_b32 v6, s0, 28
	s_nop 1
	v_writelane_b32 v6, s1, 29
	s_or_saveexec_b64 s[10:11], -1
	scratch_store_dword off, v6, s33        ; 4-byte Folded Spill
	s_mov_b64 exec, s[10:11]
.LBB2_9:
	s_or_saveexec_b64 s[10:11], -1
	scratch_load_dword v6, off, s33         ; 4-byte Folded Reload
	s_mov_b64 exec, s[10:11]
	s_waitcnt vmcnt(0)
	v_readlane_b32 s6, v6, 30
	v_readlane_b32 s7, v6, 31
	s_or_b64 exec, exec, s[6:7]
	v_readlane_b32 s2, v6, 8
	v_readlane_b32 s3, v6, 9
	v_readlane_b32 s4, v6, 26
	v_readlane_b32 s5, v6, 27
	v_readlane_b32 s0, v6, 28
	v_readlane_b32 s1, v6, 29
	s_and_b64 s[0:1], s[0:1], exec
	s_andn2_b64 s[2:3], s[2:3], exec
	s_and_b64 s[4:5], s[4:5], exec
	s_or_b64 s[2:3], s[2:3], s[4:5]
	v_writelane_b32 v6, s2, 10
	s_nop 1
	v_writelane_b32 v6, s3, 11
	v_writelane_b32 v6, s0, 12
	s_nop 1
	v_writelane_b32 v6, s1, 13
	s_or_saveexec_b64 s[10:11], -1
	scratch_store_dword off, v6, s33        ; 4-byte Folded Spill
	s_mov_b64 exec, s[10:11]
	s_branch .LBB2_13
.LBB2_10:
	s_or_saveexec_b64 s[10:11], -1
	scratch_load_dword v6, off, s33         ; 4-byte Folded Reload
	s_mov_b64 exec, s[10:11]
	s_waitcnt vmcnt(0)
	v_readlane_b32 s0, v6, 32
	v_readlane_b32 s1, v6, 33
	scratch_load_dwordx2 v[0:1], off, s33 offset:4 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[2:3], off, s33 offset:20 ; 8-byte Folded Reload
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[0:1], v[2:3], off sc1
	s_mov_b64 s[2:3], 0
	s_andn2_b64 s[0:1], s[0:1], exec
	v_writelane_b32 v6, s0, 34
	s_nop 1
	v_writelane_b32 v6, s1, 35
	s_or_saveexec_b64 s[10:11], -1
	scratch_store_dword off, v6, s33        ; 4-byte Folded Spill
	s_mov_b64 exec, s[10:11]
.LBB2_11:
	s_or_saveexec_b64 s[10:11], -1
	scratch_load_dword v6, off, s33         ; 4-byte Folded Reload
	s_mov_b64 exec, s[10:11]
	s_waitcnt vmcnt(0)
	v_readlane_b32 s0, v6, 36
	v_readlane_b32 s1, v6, 37
	s_or_b64 exec, exec, s[0:1]
	v_readlane_b32 s2, v6, 34
	v_readlane_b32 s3, v6, 35
	s_mov_b64 s[0:1], exec
	v_writelane_b32 v6, s0, 38
	s_nop 1
	v_writelane_b32 v6, s1, 39
	s_or_saveexec_b64 s[10:11], -1
	scratch_store_dword off, v6, s33        ; 4-byte Folded Spill
	s_mov_b64 exec, s[10:11]
	s_and_b64 s[0:1], s[0:1], s[2:3]
	s_mov_b64 exec, s[0:1]
	s_cbranch_execz .LBB2_19
; %bb.12:
	scratch_load_dwordx2 v[0:1], off, s33 offset:4 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[2:3], off, s33 offset:20 ; 8-byte Folded Reload
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[0:1], v[2:3], off sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	s_branch .LBB2_19
.LBB2_13:
	s_or_saveexec_b64 s[10:11], -1
	scratch_load_dword v6, off, s33         ; 4-byte Folded Reload
	s_mov_b64 exec, s[10:11]
	s_waitcnt vmcnt(0)
	v_readlane_b32 s4, v6, 14
	v_readlane_b32 s5, v6, 15
	s_or_b64 exec, exec, s[4:5]
	v_readlane_b32 s0, v6, 10
	v_readlane_b32 s1, v6, 11
	v_readlane_b32 s2, v6, 12
	v_readlane_b32 s3, v6, 13
	s_nop 0
	v_writelane_b32 v6, s2, 32
	s_nop 1
	v_writelane_b32 v6, s3, 33
	v_writelane_b32 v6, s2, 34
	s_nop 1
	v_writelane_b32 v6, s3, 35
	s_mov_b64 s[2:3], exec
	s_and_b64 s[0:1], s[2:3], s[0:1]
	s_xor_b64 s[2:3], s[0:1], s[2:3]
	v_writelane_b32 v6, s2, 36
	s_nop 1
	v_writelane_b32 v6, s3, 37
	s_or_saveexec_b64 s[10:11], -1
	scratch_store_dword off, v6, s33        ; 4-byte Folded Spill
	s_mov_b64 exec, s[10:11]
	s_mov_b64 exec, s[0:1]
	s_cbranch_execz .LBB2_11
	s_branch .LBB2_10
.LBB2_14:
	scratch_load_dwordx2 v[0:1], off, s33 offset:4 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[2:3], off, s33 offset:20 ; 8-byte Folded Reload
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[0:1], v[2:3], off sc1
	s_branch .LBB2_7
.LBB2_15:
	s_or_saveexec_b64 s[10:11], -1
	scratch_load_dword v6, off, s33         ; 4-byte Folded Reload
	s_mov_b64 exec, s[10:11]
	s_waitcnt vmcnt(0)
	v_readlane_b32 s2, v6, 18
	v_readlane_b32 s3, v6, 19
	s_or_b64 exec, exec, s[2:3]
	v_readlane_b32 s0, v6, 16
	v_readlane_b32 s1, v6, 17
	s_and_b64 s[0:1], s[0:1], exec
	v_writelane_b32 v6, s0, 4
	s_nop 1
	v_writelane_b32 v6, s1, 5
	s_or_saveexec_b64 s[10:11], -1
	scratch_store_dword off, v6, s33        ; 4-byte Folded Spill
	s_mov_b64 exec, s[10:11]
	s_branch .LBB2_17
.LBB2_16:
	scratch_load_dwordx2 v[0:1], off, s33 offset:4 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[2:3], off, s33 offset:20 ; 8-byte Folded Reload
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[0:1], v[2:3], off sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	s_branch .LBB2_5
.LBB2_17:
	s_or_saveexec_b64 s[10:11], -1
	scratch_load_dword v6, off, s33         ; 4-byte Folded Reload
	s_mov_b64 exec, s[10:11]
	s_waitcnt vmcnt(0)
	v_readlane_b32 s0, v6, 6
	v_readlane_b32 s1, v6, 7
	s_or_saveexec_b64 s[0:1], s[0:1]
	v_readlane_b32 s2, v6, 4
	v_readlane_b32 s3, v6, 5
	s_nop 0
	v_writelane_b32 v6, s2, 22
	s_nop 1
	v_writelane_b32 v6, s3, 23
	s_and_b64 s[0:1], exec, s[0:1]
	v_writelane_b32 v6, s0, 20
	s_nop 1
	v_writelane_b32 v6, s1, 21
	s_or_saveexec_b64 s[10:11], -1
	scratch_store_dword off, v6, s33        ; 4-byte Folded Spill
	s_mov_b64 exec, s[10:11]
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execz .LBB2_5
	s_branch .LBB2_16
.LBB2_18:
	s_or_saveexec_b64 s[10:11], -1
	scratch_load_dword v6, off, s33         ; 4-byte Folded Reload
	s_mov_b64 exec, s[10:11]
	scratch_load_dwordx2 v[0:1], off, s33 offset:4 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[2:3], off, s33 offset:20 ; 8-byte Folded Reload
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0) lgkmcnt(0)
	global_atomic_add_x2 v[0:1], v[2:3], off sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	s_mov_b64 s[0:1], 0
	s_xor_b64 s[0:1], exec, -1
	v_writelane_b32 v6, s0, 16
	s_nop 1
	v_writelane_b32 v6, s1, 17
	s_or_saveexec_b64 s[10:11], -1
	scratch_store_dword off, v6, s33        ; 4-byte Folded Spill
	s_mov_b64 exec, s[10:11]
	s_branch .LBB2_15
.LBB2_19:
	s_or_saveexec_b64 s[10:11], -1
	scratch_load_dword v6, off, s33         ; 4-byte Folded Reload
	s_mov_b64 exec, s[10:11]
	s_waitcnt vmcnt(0)
	v_readlane_b32 s0, v6, 38
	v_readlane_b32 s1, v6, 39
	s_or_b64 exec, exec, s[0:1]
	scratch_load_dwordx2 v[0:1], off, s33 offset:12 ; 8-byte Folded Reload
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[0:1], v[0:1], off offset:16
	s_waitcnt vmcnt(0)
	scratch_store_dwordx2 off, v[0:1], s33 offset:36 ; 8-byte Folded Spill
	s_mov_b64 s[0:1], 0
	v_cmp_ne_u64_e64 s[2:3], v[0:1], s[0:1]
	s_mov_b64 s[0:1], exec
	v_writelane_b32 v6, s0, 40
	s_nop 1
	v_writelane_b32 v6, s1, 41
	s_or_saveexec_b64 s[10:11], -1
	scratch_store_dword off, v6, s33        ; 4-byte Folded Spill
	s_mov_b64 exec, s[10:11]
	s_and_b64 s[0:1], s[0:1], s[2:3]
	s_mov_b64 exec, s[0:1]
	s_cbranch_execz .LBB2_21
; %bb.20:
	scratch_load_dwordx2 v[2:3], off, s33 offset:36 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[0:1], off, s33 offset:12 ; 8-byte Folded Reload
	s_waitcnt vmcnt(0)
	global_load_dword v0, v[0:1], off offset:24
	s_mov_b32 s0, 0
	v_mov_b32_e32 v1, 0
	s_waitcnt vmcnt(0)
	v_mov_b32_e32 v4, v0
	v_mov_b32_e32 v5, v1
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_store_dwordx2 v[2:3], v[4:5], off sc0 sc1
	s_getpc_b64 s[0:1]
	s_add_u32 s0, s0, __oclc_ISA_version@rel32@lo+4
	s_addc_u32 s1, s1, __oclc_ISA_version@rel32@hi+12
	s_load_dword s0, s[0:1], 0x0
	s_mov_b32 s1, 0x2af8
	s_waitcnt lgkmcnt(0)
	s_cmp_lt_u32 s0, s1
	s_mov_b32 s1, 0xffffff
	s_mov_b32 s2, 0x7fffff
	s_cselect_b32 s2, s2, s1
	s_mov_b32 s3, 0x2710
	s_cmp_lt_u32 s0, s3
	s_cselect_b32 s1, s1, s2
	s_mov_b32 s2, 0x2328
	s_cmp_lt_i32 s0, s2
	s_mov_b32 s0, 0xff
	s_cselect_b32 s0, s0, s1
	v_and_b32_e64 v0, s0, v0
	s_nop 0
	v_readfirstlane_b32 s0, v0
	s_mov_b32 m0, s0
	s_nop 0
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB2_21:
	s_or_saveexec_b64 s[10:11], -1
	scratch_load_dword v6, off, s33         ; 4-byte Folded Reload
	s_mov_b64 exec, s[10:11]
	s_waitcnt vmcnt(0)
	v_readlane_b32 s0, v6, 40
	v_readlane_b32 s1, v6, 41
	s_or_b64 exec, exec, s[0:1]
	s_mov_b32 s32, s33
	s_xor_saveexec_b64 s[0:1], -1
	scratch_load_dword v6, off, s33 offset:44 ; 4-byte Folded Reload
	s_mov_b64 exec, s[0:1]
	s_mov_b32 s33, s12
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_setpc_b64 s[30:31]
.Lfunc_end2:
	.size	__ockl_hsa_signal_add, .Lfunc_end2-__ockl_hsa_signal_add
                                        ; -- End function
	.set .L__ockl_hsa_signal_add.num_vgpr, 7
	.set .L__ockl_hsa_signal_add.num_agpr, 0
	.set .L__ockl_hsa_signal_add.numbered_sgpr, 34
	.set .L__ockl_hsa_signal_add.num_named_barrier, 0
	.set .L__ockl_hsa_signal_add.private_seg_size, 52
	.set .L__ockl_hsa_signal_add.uses_vcc, 0
	.set .L__ockl_hsa_signal_add.uses_flat_scratch, 0
	.set .L__ockl_hsa_signal_add.has_dyn_sized_stack, 0
	.set .L__ockl_hsa_signal_add.has_recursion, 0
	.set .L__ockl_hsa_signal_add.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Function info:
; codeLenInByte = 2564
; TotalNumSgprs: 40
; NumVgprs: 7
; NumAgprs: 0
; TotalNumVgprs: 7
; ScratchSize: 52
; MemoryBound: 0
	.text
	.p2align	2                               ; -- Begin function __ockl_hostcall_internal
	.type	__ockl_hostcall_internal,@function
__ockl_hostcall_internal:               ; @__ockl_hostcall_internal
; %bb.0:
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
	s_mov_b32 s13, s33
	s_mov_b32 s33, s32
	s_xor_saveexec_b64 s[0:1], -1
	scratch_store_dword off, v21, s33 offset:204 ; 4-byte Folded Spill
	scratch_store_dword off, v22, s33 offset:208 ; 4-byte Folded Spill
	s_mov_b64 exec, s[0:1]
	s_add_i32 s32, s32, 0xe0
	v_writelane_b32 v21, s30, 0
	s_nop 1
	v_writelane_b32 v21, s31, 1
	v_accvgpr_write_b32 a0, v18             ;  Reload Reuse
	v_accvgpr_write_b32 a1, v17             ;  Reload Reuse
	v_mov_b32_e32 v17, v16
	v_accvgpr_read_b32 v16, a1              ;  Reload Reuse
	v_accvgpr_write_b32 a2, v17             ;  Reload Reuse
	v_mov_b32_e32 v17, v15
	v_accvgpr_read_b32 v15, a0              ;  Reload Reuse
	v_accvgpr_write_b32 a3, v17             ;  Reload Reuse
	v_mov_b32_e32 v17, v14
	v_accvgpr_read_b32 v14, a3              ;  Reload Reuse
	v_accvgpr_write_b32 a4, v17             ;  Reload Reuse
	v_mov_b32_e32 v17, v13
	v_accvgpr_read_b32 v13, a2              ;  Reload Reuse
	v_accvgpr_write_b32 a5, v17             ;  Reload Reuse
	v_mov_b32_e32 v17, v12
	v_accvgpr_read_b32 v12, a5              ;  Reload Reuse
	v_accvgpr_write_b32 a6, v17             ;  Reload Reuse
	v_mov_b32_e32 v17, v11
	v_accvgpr_read_b32 v11, a4              ;  Reload Reuse
	v_accvgpr_write_b32 a7, v17             ;  Reload Reuse
	v_mov_b32_e32 v17, v10
	v_accvgpr_read_b32 v10, a7              ;  Reload Reuse
	v_accvgpr_write_b32 a8, v17             ;  Reload Reuse
	v_mov_b32_e32 v17, v9
	v_accvgpr_read_b32 v9, a6               ;  Reload Reuse
	v_accvgpr_write_b32 a9, v17             ;  Reload Reuse
	v_mov_b32_e32 v17, v8
	v_accvgpr_read_b32 v8, a9               ;  Reload Reuse
	v_accvgpr_write_b32 a10, v17            ;  Reload Reuse
	v_mov_b32_e32 v17, v7
	v_accvgpr_read_b32 v7, a8               ;  Reload Reuse
	v_accvgpr_write_b32 a11, v17            ;  Reload Reuse
	v_mov_b32_e32 v17, v6
	v_accvgpr_read_b32 v6, a11              ;  Reload Reuse
	v_accvgpr_write_b32 a12, v17            ;  Reload Reuse
	v_mov_b32_e32 v17, v5
	v_accvgpr_read_b32 v5, a10              ;  Reload Reuse
	v_accvgpr_write_b32 a13, v17            ;  Reload Reuse
	v_mov_b32_e32 v17, v4
	v_accvgpr_read_b32 v4, a13              ;  Reload Reuse
	v_accvgpr_write_b32 a14, v17            ;  Reload Reuse
	v_mov_b32_e32 v17, v3
	v_accvgpr_read_b32 v3, a14              ;  Reload Reuse
	v_accvgpr_write_b32 a15, v17            ;  Reload Reuse
	v_accvgpr_write_b32 a16, v2             ;  Reload Reuse
	v_mov_b32_e32 v18, v1
	v_accvgpr_read_b32 v1, a12              ;  Reload Reuse
	v_mov_b32_e32 v2, v0
	v_accvgpr_read_b32 v0, a15              ;  Reload Reuse
                                        ; kill: def $vgpr16 killed $vgpr16 def $vgpr16_vgpr17 killed $exec
	v_mov_b32_e32 v17, v15
                                        ; kill: def $vgpr14 killed $vgpr14 def $vgpr14_vgpr15 killed $exec
	v_mov_b32_e32 v15, v13
                                        ; kill: def $vgpr12 killed $vgpr12 def $vgpr12_vgpr13 killed $exec
	v_mov_b32_e32 v13, v11
                                        ; kill: def $vgpr10 killed $vgpr10 def $vgpr10_vgpr11 killed $exec
	v_mov_b32_e32 v11, v9
                                        ; kill: def $vgpr8 killed $vgpr8 def $vgpr8_vgpr9 killed $exec
	v_mov_b32_e32 v9, v7
                                        ; kill: def $vgpr6 killed $vgpr6 def $vgpr6_vgpr7 killed $exec
	v_mov_b32_e32 v7, v5
                                        ; kill: def $vgpr4 killed $vgpr4 def $vgpr4_vgpr5 killed $exec
	v_mov_b32_e32 v5, v1
                                        ; kill: def $vgpr0 killed $vgpr0 def $vgpr0_vgpr1 killed $exec
	v_mov_b32_e32 v1, v3
                                        ; kill: def $vgpr2 killed $vgpr2 def $vgpr2_vgpr3 killed $exec
	v_mov_b32_e32 v3, v18
	v_accvgpr_write_b32 a17, v17            ;  Reload Reuse
	v_accvgpr_write_b32 a18, v16            ;  Reload Reuse
	v_accvgpr_write_b32 a19, v15            ;  Reload Reuse
	v_accvgpr_write_b32 a20, v14            ;  Reload Reuse
	v_accvgpr_write_b32 a21, v13            ;  Reload Reuse
	v_accvgpr_write_b32 a22, v12            ;  Reload Reuse
	v_accvgpr_write_b32 a23, v11            ;  Reload Reuse
	v_accvgpr_write_b32 a24, v10            ;  Reload Reuse
	v_accvgpr_write_b32 a25, v9             ;  Reload Reuse
	v_accvgpr_write_b32 a26, v8             ;  Reload Reuse
	v_accvgpr_write_b32 a27, v7             ;  Reload Reuse
	v_accvgpr_write_b32 a28, v6             ;  Reload Reuse
	v_accvgpr_write_b32 a29, v5             ;  Reload Reuse
	v_accvgpr_write_b32 a30, v4             ;  Reload Reuse
	v_accvgpr_write_b32 a31, v1             ;  Reload Reuse
	scratch_store_dword off, v0, s33 offset:24 ; 4-byte Folded Spill
	s_mov_b32 s1, 0
	s_mov_b32 s0, -1
	v_mov_b32_e32 v0, s1
	v_mbcnt_lo_u32_b32 v0, s0, v0
	v_mbcnt_hi_u32_b32 v0, s0, v0
	scratch_store_dword off, v0, s33 offset:20 ; 4-byte Folded Spill
	v_readfirstlane_b32 s0, v0
	scratch_store_dwordx2 off, v[2:3], s33 offset:12 ; 8-byte Folded Spill
	s_nop 0
	v_cmp_eq_u32_e64 s[2:3], v0, s0
	s_mov_b64 s[0:1], s[2:3]
                                        ; implicit-def: $vgpr22 : SGPR spill to VGPR lane
	v_writelane_b32 v22, s0, 0
	s_nop 1
	v_writelane_b32 v22, s1, 1
	v_mov_b64_e32 v[0:1], 0
	scratch_store_dwordx2 off, v[0:1], s33 offset:4 ; 8-byte Folded Spill
	s_mov_b64 s[0:1], exec
	v_writelane_b32 v22, s0, 2
	s_nop 1
	v_writelane_b32 v22, s1, 3
	s_or_saveexec_b64 s[14:15], -1
	scratch_store_dword off, v22, s33       ; 4-byte Folded Spill
	s_mov_b64 exec, s[14:15]
	s_and_b64 s[0:1], s[0:1], s[2:3]
	s_mov_b64 exec, s[0:1]
	s_cbranch_execz .LBB3_6
; %bb.1:
	s_or_saveexec_b64 s[14:15], -1
	scratch_load_dword v22, off, s33        ; 4-byte Folded Reload
	s_mov_b64 exec, s[14:15]
	scratch_load_dwordx2 v[0:1], off, s33 offset:12 ; 8-byte Folded Reload
	s_mov_b64 s[0:1], 24
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[0:1], 0, s[0:1]
	scratch_store_dwordx2 off, v[2:3], s33 offset:56 ; 8-byte Folded Spill
	global_load_dwordx2 v[2:3], v[0:1], off offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	s_mov_b64 s[0:1], 40
	v_lshl_add_u64 v[4:5], v[0:1], 0, s[0:1]
	scratch_store_dwordx2 off, v[4:5], s33 offset:48 ; 8-byte Folded Spill
	global_load_dwordx2 v[4:5], v[0:1], off
	s_nop 0
	global_load_dwordx2 v[6:7], v[0:1], off offset:40
	v_mov_b32_e32 v8, v3
	s_waitcnt vmcnt(0)
	v_mov_b32_e32 v9, v7
	v_and_b32_e64 v10, v9, v8
	v_mov_b32_e32 v9, v2
                                        ; kill: def $vgpr6 killed $vgpr6 killed $vgpr6_vgpr7 killed $exec
	v_and_b32_e64 v6, v6, v9
                                        ; kill: def $vgpr6 killed $vgpr6 def $vgpr6_vgpr7 killed $exec
	v_mov_b32_e32 v7, v10
	v_mov_b32_e32 v10, v6
	s_mov_b32 s1, 24
	v_mad_u64_u32 v[12:13], s[2:3], v10, s1, 0
	v_mov_b32_e32 v10, v13
                                        ; implicit-def: $sgpr0
                                        ; implicit-def: $sgpr2
	v_mov_b32_e32 v14, s0
                                        ; kill: def $vgpr10 killed $vgpr10 def $vgpr10_vgpr11 killed $exec
	v_mov_b32_e32 v11, v14
	s_mov_b32 s0, 32
	v_lshrrev_b64 v[6:7], s0, v[6:7]
                                        ; kill: def $vgpr6 killed $vgpr6 killed $vgpr6_vgpr7 killed $exec
	v_mad_u64_u32 v[6:7], s[2:3], v6, s1, v[10:11]
                                        ; kill: def $vgpr6 killed $vgpr6 killed $vgpr6_vgpr7 killed $exec
                                        ; implicit-def: $sgpr1
                                        ; implicit-def: $sgpr2
	v_mov_b32_e32 v10, s1
                                        ; kill: def $vgpr6 killed $vgpr6 def $vgpr6_vgpr7 killed $exec
	v_mov_b32_e32 v7, v10
	v_lshlrev_b64 v[6:7], s0, v[6:7]
	v_mov_b32_e32 v11, v7
                                        ; kill: def $vgpr12 killed $vgpr12 killed $vgpr12_vgpr13 killed $exec
	s_mov_b32 s0, 0
	v_mov_b32_e32 v10, 0
                                        ; kill: def $vgpr12 killed $vgpr12 def $vgpr12_vgpr13 killed $exec
	v_mov_b32_e32 v13, v10
	v_mov_b32_e32 v10, v13
	v_or_b32_e64 v10, v10, v11
	v_mov_b32_e32 v7, v6
	v_mov_b32_e32 v6, v12
	v_or_b32_e64 v6, v6, v7
                                        ; kill: def $vgpr6 killed $vgpr6 def $vgpr6_vgpr7 killed $exec
	v_mov_b32_e32 v7, v10
	v_lshl_add_u64 v[4:5], v[4:5], 0, v[6:7]
	global_load_dwordx2 v[4:5], v[4:5], off sc0 sc1
	s_waitcnt vmcnt(0)
	v_mov_b32_e32 v10, v5
                                        ; kill: def $vgpr4 killed $vgpr4 killed $vgpr4_vgpr5 killed $exec
                                        ; kill: def $vgpr4 killed $vgpr4 def $vgpr4_vgpr5_vgpr6_vgpr7 killed $exec
	v_mov_b32_e32 v5, v10
	v_mov_b32_e32 v6, v9
	v_mov_b32_e32 v7, v8
	global_atomic_cmpswap_x2 v[0:1], v[0:1], v[4:7], off offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_ne_u64_e64 s[2:3], v[0:1], v[2:3]
	s_mov_b64 s[0:1], 0
	v_writelane_b32 v22, s0, 4
	s_nop 1
	v_writelane_b32 v22, s1, 5
	v_mov_b64_e32 v[2:3], v[0:1]
	scratch_store_dwordx2 off, v[2:3], s33 offset:40 ; 8-byte Folded Spill
	scratch_store_dwordx2 off, v[0:1], s33 offset:32 ; 8-byte Folded Spill
	s_mov_b64 s[0:1], exec
	v_writelane_b32 v22, s0, 6
	s_nop 1
	v_writelane_b32 v22, s1, 7
	s_or_saveexec_b64 s[14:15], -1
	scratch_store_dword off, v22, s33       ; 4-byte Folded Spill
	s_mov_b64 exec, s[14:15]
	s_and_b64 s[0:1], s[0:1], s[2:3]
	s_mov_b64 exec, s[0:1]
	s_cbranch_execz .LBB3_5
.LBB3_2:                                ; =>This Inner Loop Header: Depth=1
	s_or_saveexec_b64 s[14:15], -1
	scratch_load_dword v22, off, s33        ; 4-byte Folded Reload
	s_mov_b64 exec, s[14:15]
	s_waitcnt vmcnt(0)
	v_readlane_b32 s2, v22, 4
	v_readlane_b32 s3, v22, 5
	scratch_load_dwordx2 v[2:3], off, s33 offset:40 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[0:1], off, s33 offset:56 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[6:7], off, s33 offset:48 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[4:5], off, s33 offset:12 ; 8-byte Folded Reload
	s_sleep 1
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[4:5], v[4:5], off
	s_nop 0
	global_load_dwordx2 v[6:7], v[6:7], off
	v_mov_b32_e32 v8, v3
	s_waitcnt vmcnt(0)
	v_mov_b32_e32 v9, v7
	v_and_b32_e64 v10, v9, v8
	v_mov_b32_e32 v9, v2
                                        ; kill: def $vgpr6 killed $vgpr6 killed $vgpr6_vgpr7 killed $exec
	v_and_b32_e64 v6, v6, v9
                                        ; kill: def $vgpr6 killed $vgpr6 def $vgpr6_vgpr7 killed $exec
	v_mov_b32_e32 v7, v10
	v_mov_b32_e32 v10, v6
	s_mov_b32 s1, 24
	v_mad_u64_u32 v[12:13], s[4:5], v10, s1, 0
	v_mov_b32_e32 v10, v13
                                        ; implicit-def: $sgpr0
                                        ; implicit-def: $sgpr4
	v_mov_b32_e32 v14, s0
                                        ; kill: def $vgpr10 killed $vgpr10 def $vgpr10_vgpr11 killed $exec
	v_mov_b32_e32 v11, v14
	s_mov_b32 s0, 32
	v_lshrrev_b64 v[6:7], s0, v[6:7]
                                        ; kill: def $vgpr6 killed $vgpr6 killed $vgpr6_vgpr7 killed $exec
	v_mad_u64_u32 v[6:7], s[4:5], v6, s1, v[10:11]
                                        ; kill: def $vgpr6 killed $vgpr6 killed $vgpr6_vgpr7 killed $exec
                                        ; implicit-def: $sgpr1
                                        ; implicit-def: $sgpr4
	v_mov_b32_e32 v10, s1
                                        ; kill: def $vgpr6 killed $vgpr6 def $vgpr6_vgpr7 killed $exec
	v_mov_b32_e32 v7, v10
	v_lshlrev_b64 v[6:7], s0, v[6:7]
	v_mov_b32_e32 v11, v7
                                        ; kill: def $vgpr12 killed $vgpr12 killed $vgpr12_vgpr13 killed $exec
	s_mov_b32 s0, 0
	v_mov_b32_e32 v10, 0
                                        ; kill: def $vgpr12 killed $vgpr12 def $vgpr12_vgpr13 killed $exec
	v_mov_b32_e32 v13, v10
	v_mov_b32_e32 v10, v13
	v_or_b32_e64 v10, v10, v11
	v_mov_b32_e32 v7, v6
	v_mov_b32_e32 v6, v12
	v_or_b32_e64 v6, v6, v7
                                        ; kill: def $vgpr6 killed $vgpr6 def $vgpr6_vgpr7 killed $exec
	v_mov_b32_e32 v7, v10
	v_lshl_add_u64 v[4:5], v[4:5], 0, v[6:7]
	global_load_dwordx2 v[4:5], v[4:5], off sc0 sc1
	s_waitcnt vmcnt(0)
	v_mov_b32_e32 v10, v5
                                        ; kill: def $vgpr4 killed $vgpr4 killed $vgpr4_vgpr5 killed $exec
                                        ; kill: def $vgpr4 killed $vgpr4 def $vgpr4_vgpr5_vgpr6_vgpr7 killed $exec
	v_mov_b32_e32 v5, v10
	v_mov_b32_e32 v6, v9
	v_mov_b32_e32 v7, v8
	global_atomic_cmpswap_x2 v[0:1], v[0:1], v[4:7], off sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	v_cmp_eq_u64_e64 s[0:1], v[0:1], v[2:3]
	s_or_b64 s[0:1], s[0:1], s[2:3]
	s_mov_b64 s[2:3], s[0:1]
	v_writelane_b32 v22, s2, 4
	s_nop 1
	v_writelane_b32 v22, s3, 5
	v_mov_b64_e32 v[2:3], v[0:1]
	scratch_store_dwordx2 off, v[2:3], s33 offset:40 ; 8-byte Folded Spill
	scratch_store_dwordx2 off, v[0:1], s33 offset:64 ; 8-byte Folded Spill
	s_mov_b64 s[2:3], s[0:1]
	v_writelane_b32 v22, s2, 8
	s_nop 1
	v_writelane_b32 v22, s3, 9
	s_or_saveexec_b64 s[14:15], -1
	scratch_store_dword off, v22, s33       ; 4-byte Folded Spill
	s_mov_b64 exec, s[14:15]
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB3_2
; %bb.3:
	s_or_saveexec_b64 s[14:15], -1
	scratch_load_dword v22, off, s33        ; 4-byte Folded Reload
	s_mov_b64 exec, s[14:15]
	s_waitcnt vmcnt(0)
	v_readlane_b32 s0, v22, 8
	v_readlane_b32 s1, v22, 9
	s_or_b64 exec, exec, s[0:1]
; %bb.4:
	scratch_load_dwordx2 v[0:1], off, s33 offset:64 ; 8-byte Folded Reload
	s_waitcnt vmcnt(0)
	scratch_store_dwordx2 off, v[0:1], s33 offset:32 ; 8-byte Folded Spill
.LBB3_5:
	s_or_saveexec_b64 s[14:15], -1
	scratch_load_dword v22, off, s33        ; 4-byte Folded Reload
	s_mov_b64 exec, s[14:15]
	s_waitcnt vmcnt(0)
	v_readlane_b32 s0, v22, 6
	v_readlane_b32 s1, v22, 7
	s_or_b64 exec, exec, s[0:1]
	scratch_load_dwordx2 v[0:1], off, s33 offset:32 ; 8-byte Folded Reload
	s_waitcnt vmcnt(0)
	scratch_store_dwordx2 off, v[0:1], s33 offset:4 ; 8-byte Folded Spill
.LBB3_6:
	s_or_saveexec_b64 s[14:15], -1
	scratch_load_dword v22, off, s33        ; 4-byte Folded Reload
	s_mov_b64 exec, s[14:15]
	s_waitcnt vmcnt(0)
	v_readlane_b32 s0, v22, 2
	v_readlane_b32 s1, v22, 3
	s_or_b64 exec, exec, s[0:1]
	v_readlane_b32 s2, v22, 0
	v_readlane_b32 s3, v22, 1
	scratch_load_dwordx2 v[0:1], off, s33 offset:12 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[2:3], off, s33 offset:4 ; 8-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_mov_b32_e32 v4, v3
	s_nop 0
	v_readfirstlane_b32 s4, v4
                                        ; kill: def $vgpr2 killed $vgpr2 killed $vgpr2_vgpr3 killed $exec
	v_readfirstlane_b32 s0, v2
                                        ; kill: def $sgpr0 killed $sgpr0 def $sgpr0_sgpr1
	s_mov_b32 s1, s4
	s_mov_b64 s[4:5], s[0:1]
	v_writelane_b32 v22, s4, 10
	s_nop 1
	v_writelane_b32 v22, s5, 11
	global_load_dwordx2 v[4:5], v[0:1], off
	s_waitcnt vmcnt(0)
	scratch_store_dwordx2 off, v[4:5], s33 offset:96 ; 8-byte Folded Spill
	s_mov_b64 s[4:5], 40
	v_lshl_add_u64 v[2:3], v[0:1], 0, s[4:5]
	scratch_store_dwordx2 off, v[2:3], s33 offset:88 ; 8-byte Folded Spill
	global_load_dwordx2 v[2:3], v[0:1], off offset:40
	s_mov_b32 s4, s1
	s_waitcnt vmcnt(0)
	v_mov_b32_e32 v6, v3
	v_and_b32_e64 v6, v6, s4
                                        ; kill: def $sgpr0 killed $sgpr0 killed $sgpr0_sgpr1
                                        ; kill: def $vgpr2 killed $vgpr2 killed $vgpr2_vgpr3 killed $exec
	v_and_b32_e64 v2, v2, s0
                                        ; kill: def $vgpr2 killed $vgpr2 def $vgpr2_vgpr3 killed $exec
	v_mov_b32_e32 v3, v6
	v_mov_b32_e32 v6, v2
	s_mov_b32 s1, 24
	v_mad_u64_u32 v[10:11], s[4:5], v6, s1, 0
	v_mov_b32_e32 v8, v11
                                        ; implicit-def: $sgpr0
                                        ; implicit-def: $sgpr4
	v_mov_b32_e32 v6, s0
                                        ; kill: def $vgpr8 killed $vgpr8 def $vgpr8_vgpr9 killed $exec
	v_mov_b32_e32 v9, v6
	s_mov_b32 s0, 32
	v_lshrrev_b64 v[6:7], s0, v[2:3]
                                        ; kill: def $vgpr6 killed $vgpr6 killed $vgpr6_vgpr7 killed $exec
	v_mad_u64_u32 v[6:7], s[4:5], v6, s1, v[8:9]
                                        ; kill: def $vgpr6 killed $vgpr6 killed $vgpr6_vgpr7 killed $exec
                                        ; implicit-def: $sgpr1
                                        ; implicit-def: $sgpr4
	v_mov_b32_e32 v8, s1
                                        ; kill: def $vgpr6 killed $vgpr6 def $vgpr6_vgpr7 killed $exec
	v_mov_b32_e32 v7, v8
	v_lshlrev_b64 v[6:7], s0, v[6:7]
	v_mov_b32_e32 v9, v7
                                        ; kill: def $vgpr10 killed $vgpr10 killed $vgpr10_vgpr11 killed $exec
	s_mov_b32 s0, 0
	v_mov_b32_e32 v8, 0
                                        ; kill: def $vgpr10 killed $vgpr10 def $vgpr10_vgpr11 killed $exec
	v_mov_b32_e32 v11, v8
	v_mov_b32_e32 v8, v11
	v_or_b32_e64 v8, v8, v9
	v_mov_b32_e32 v7, v6
	v_mov_b32_e32 v6, v10
	v_or_b32_e64 v6, v6, v7
                                        ; kill: def $vgpr6 killed $vgpr6 def $vgpr6_vgpr7 killed $exec
	v_mov_b32_e32 v7, v8
	v_lshl_add_u64 v[4:5], v[4:5], 0, v[6:7]
	scratch_store_dwordx2 off, v[4:5], s33 offset:80 ; 8-byte Folded Spill
	global_load_dwordx2 v[0:1], v[0:1], off offset:8
	s_mov_b32 s0, 12
	v_lshlrev_b64 v[2:3], s0, v[2:3]
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[0:1], v[0:1], 0, v[2:3]
	scratch_store_dwordx2 off, v[0:1], s33 offset:72 ; 8-byte Folded Spill
	s_mov_b64 s[0:1], exec
	v_writelane_b32 v22, s0, 12
	s_nop 1
	v_writelane_b32 v22, s1, 13
	s_mov_b64 s[0:1], exec
	v_writelane_b32 v22, s0, 14
	s_nop 1
	v_writelane_b32 v22, s1, 15
	s_or_saveexec_b64 s[14:15], -1
	scratch_store_dword off, v22, s33       ; 4-byte Folded Spill
	s_mov_b64 exec, s[14:15]
	s_and_b64 s[0:1], s[0:1], s[2:3]
	s_mov_b64 exec, s[0:1]
	s_cbranch_execz .LBB3_8
; %bb.7:
	s_or_saveexec_b64 s[14:15], -1
	scratch_load_dword v22, off, s33        ; 4-byte Folded Reload
	s_mov_b64 exec, s[14:15]
	s_waitcnt vmcnt(0)
	v_readlane_b32 s0, v22, 12
	v_readlane_b32 s1, v22, 13
	scratch_load_dwordx2 v[0:1], off, s33 offset:80 ; 8-byte Folded Reload
	v_accvgpr_read_b32 v2, a16              ;  Reload Reuse
	s_waitcnt vmcnt(0)
	global_store_dword v[0:1], v2, off offset:16
	v_mov_b64_e32 v[2:3], s[0:1]
	global_store_dwordx2 v[0:1], v[2:3], off offset:8
	v_mov_b32_e32 v2, 1
	global_store_dword v[0:1], v2, off offset:20
.LBB3_8:
	s_or_saveexec_b64 s[14:15], -1
	scratch_load_dword v22, off, s33        ; 4-byte Folded Reload
	s_mov_b64 exec, s[14:15]
	s_waitcnt vmcnt(0)
	v_readlane_b32 s0, v22, 14
	v_readlane_b32 s1, v22, 15
	s_or_b64 exec, exec, s[0:1]
	v_readlane_b32 s2, v22, 0
	v_readlane_b32 s3, v22, 1
	v_accvgpr_read_b32 v3, a17              ;  Reload Reuse
	v_accvgpr_read_b32 v2, a18              ;  Reload Reuse
	v_accvgpr_read_b32 v5, a19              ;  Reload Reuse
	v_accvgpr_read_b32 v4, a20              ;  Reload Reuse
	v_accvgpr_read_b32 v7, a21              ;  Reload Reuse
	v_accvgpr_read_b32 v6, a22              ;  Reload Reuse
	v_accvgpr_read_b32 v9, a23              ;  Reload Reuse
	v_accvgpr_read_b32 v8, a24              ;  Reload Reuse
	v_accvgpr_read_b32 v11, a25             ;  Reload Reuse
	v_accvgpr_read_b32 v10, a26             ;  Reload Reuse
	v_accvgpr_read_b32 v13, a27             ;  Reload Reuse
	v_accvgpr_read_b32 v12, a28             ;  Reload Reuse
	v_accvgpr_read_b32 v15, a29             ;  Reload Reuse
	v_accvgpr_read_b32 v14, a30             ;  Reload Reuse
	v_accvgpr_read_b32 v17, a31             ;  Reload Reuse
	scratch_load_dword v16, off, s33 offset:24 ; 4-byte Folded Reload
	scratch_load_dwordx2 v[0:1], off, s33 offset:72 ; 8-byte Folded Reload
	scratch_load_dword v18, off, s33 offset:20 ; 4-byte Folded Reload
	s_mov_b32 s0, 0
	v_mov_b32_e32 v20, 0
                                        ; kill: def $vgpr18 killed $vgpr18 def $vgpr18_vgpr19 killed $exec
	v_mov_b32_e32 v19, v20
	s_mov_b32 s0, 6
	s_waitcnt vmcnt(0)
	v_lshlrev_b64 v[18:19], s0, v[18:19]
	v_lshl_add_u64 v[0:1], v[0:1], 0, v[18:19]
	scratch_store_dwordx2 off, v[0:1], s33 offset:112 ; 8-byte Folded Spill
	global_store_dwordx2 v[0:1], v[16:17], off
	s_mov_b64 s[0:1], 8
	v_lshl_add_u64 v[16:17], v[0:1], 0, s[0:1]
	scratch_store_dwordx2 off, v[16:17], s33 offset:104 ; 8-byte Folded Spill
	global_store_dwordx2 v[0:1], v[14:15], off offset:8
	global_store_dwordx2 v[0:1], v[12:13], off offset:16
	global_store_dwordx2 v[0:1], v[10:11], off offset:24
	global_store_dwordx2 v[0:1], v[8:9], off offset:32
	global_store_dwordx2 v[0:1], v[6:7], off offset:40
	global_store_dwordx2 v[0:1], v[4:5], off offset:48
	global_store_dwordx2 v[0:1], v[2:3], off offset:56
	s_mov_b64 s[0:1], exec
	v_writelane_b32 v22, s0, 16
	s_nop 1
	v_writelane_b32 v22, s1, 17
	s_or_saveexec_b64 s[14:15], -1
	scratch_store_dword off, v22, s33       ; 4-byte Folded Spill
	s_mov_b64 exec, s[14:15]
	s_and_b64 s[0:1], s[0:1], s[2:3]
	s_mov_b64 exec, s[0:1]
	s_cbranch_execz .LBB3_13
; %bb.9:
	s_or_saveexec_b64 s[14:15], -1
	scratch_load_dword v22, off, s33        ; 4-byte Folded Reload
	s_mov_b64 exec, s[14:15]
	s_waitcnt vmcnt(0)
	v_readlane_b32 s2, v22, 10
	v_readlane_b32 s3, v22, 11
	scratch_load_dwordx2 v[0:1], off, s33 offset:12 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[4:5], off, s33 offset:96 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[6:7], off, s33 offset:88 ; 8-byte Folded Reload
	s_mov_b64 s[0:1], 32
	s_waitcnt vmcnt(2)
	v_lshl_add_u64 v[2:3], v[0:1], 0, s[0:1]
	scratch_store_dwordx2 off, v[2:3], s33 offset:136 ; 8-byte Folded Spill
	global_load_dwordx2 v[2:3], v[0:1], off offset:32 sc0 sc1
	s_waitcnt vmcnt(2)
	global_load_dwordx2 v[6:7], v[6:7], off
	s_mov_b32 s0, s3
	s_waitcnt vmcnt(0)
	v_mov_b32_e32 v8, v7
	v_and_b32_e64 v8, v8, s0
	s_mov_b32 s1, s2
                                        ; kill: def $vgpr6 killed $vgpr6 killed $vgpr6_vgpr7 killed $exec
	v_and_b32_e64 v6, v6, s1
                                        ; kill: def $vgpr6 killed $vgpr6 def $vgpr6_vgpr7 killed $exec
	v_mov_b32_e32 v7, v8
	v_mov_b32_e32 v8, v6
	s_mov_b32 s3, 24
	v_mad_u64_u32 v[10:11], s[4:5], v8, s3, 0
	v_mov_b32_e32 v8, v11
                                        ; implicit-def: $sgpr2
                                        ; implicit-def: $sgpr4
	v_mov_b32_e32 v12, s2
                                        ; kill: def $vgpr8 killed $vgpr8 def $vgpr8_vgpr9 killed $exec
	v_mov_b32_e32 v9, v12
	s_mov_b32 s2, 32
	v_lshrrev_b64 v[6:7], s2, v[6:7]
                                        ; kill: def $vgpr6 killed $vgpr6 killed $vgpr6_vgpr7 killed $exec
	v_mad_u64_u32 v[6:7], s[4:5], v6, s3, v[8:9]
                                        ; kill: def $vgpr6 killed $vgpr6 killed $vgpr6_vgpr7 killed $exec
                                        ; implicit-def: $sgpr3
                                        ; implicit-def: $sgpr4
	v_mov_b32_e32 v8, s3
                                        ; kill: def $vgpr6 killed $vgpr6 def $vgpr6_vgpr7 killed $exec
	v_mov_b32_e32 v7, v8
	v_lshlrev_b64 v[6:7], s2, v[6:7]
	v_mov_b32_e32 v9, v7
                                        ; kill: def $vgpr10 killed $vgpr10 killed $vgpr10_vgpr11 killed $exec
	s_mov_b32 s2, 0
	v_mov_b32_e32 v8, 0
                                        ; kill: def $vgpr10 killed $vgpr10 def $vgpr10_vgpr11 killed $exec
	v_mov_b32_e32 v11, v8
	v_mov_b32_e32 v8, v11
	v_or_b32_e64 v8, v8, v9
	v_mov_b32_e32 v7, v6
	v_mov_b32_e32 v6, v10
	v_or_b32_e64 v6, v6, v7
                                        ; kill: def $vgpr6 killed $vgpr6 def $vgpr6_vgpr7 killed $exec
	v_mov_b32_e32 v7, v8
	v_lshl_add_u64 v[4:5], v[4:5], 0, v[6:7]
	scratch_store_dwordx2 off, v[4:5], s33 offset:128 ; 8-byte Folded Spill
	global_store_dwordx2 v[4:5], v[2:3], off
	v_mov_b32_e32 v8, v3
	v_mov_b32_e32 v9, v2
	v_mov_b32_e32 v4, s1
	v_mov_b32_e32 v10, s0
                                        ; kill: def $vgpr4 killed $vgpr4 def $vgpr4_vgpr5_vgpr6_vgpr7 killed $exec
	v_mov_b32_e32 v5, v10
	v_mov_b32_e32 v6, v9
	v_mov_b32_e32 v7, v8
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[0:1], v[0:1], v[4:7], off offset:32 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e64 s[4:5], v[0:1], v[2:3]
	v_cmp_ne_u64_e64 s[2:3], v[0:1], v[2:3]
	s_mov_b64 s[0:1], 0
	v_writelane_b32 v22, s4, 18
	s_nop 1
	v_writelane_b32 v22, s5, 19
	v_writelane_b32 v22, s0, 20
	s_nop 1
	v_writelane_b32 v22, s1, 21
	scratch_store_dwordx2 off, v[0:1], s33 offset:120 ; 8-byte Folded Spill
	s_mov_b64 s[0:1], exec
	v_writelane_b32 v22, s0, 22
	s_nop 1
	v_writelane_b32 v22, s1, 23
	s_or_saveexec_b64 s[14:15], -1
	scratch_store_dword off, v22, s33       ; 4-byte Folded Spill
	s_mov_b64 exec, s[14:15]
	s_and_b64 s[0:1], s[0:1], s[2:3]
	s_mov_b64 exec, s[0:1]
	s_cbranch_execz .LBB3_14
.LBB3_10:                               ; =>This Inner Loop Header: Depth=1
	s_or_saveexec_b64 s[14:15], -1
	scratch_load_dword v22, off, s33        ; 4-byte Folded Reload
	s_mov_b64 exec, s[14:15]
	s_waitcnt vmcnt(0)
	v_readlane_b32 s0, v22, 20
	v_readlane_b32 s1, v22, 21
	v_readlane_b32 s2, v22, 18
	v_readlane_b32 s3, v22, 19
	v_readlane_b32 s4, v22, 10
	v_readlane_b32 s5, v22, 11
	scratch_load_dwordx2 v[2:3], off, s33 offset:120 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[0:1], off, s33 offset:136 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[4:5], off, s33 offset:128 ; 8-byte Folded Reload
	s_sleep 1
	s_waitcnt vmcnt(0)
	global_store_dwordx2 v[4:5], v[2:3], off
	v_mov_b32_e32 v8, v3
	v_mov_b32_e32 v9, v2
	s_mov_b32 s2, s5
	s_mov_b32 s3, s4
	v_mov_b32_e32 v4, s3
	v_mov_b32_e32 v10, s2
                                        ; kill: def $vgpr4 killed $vgpr4 def $vgpr4_vgpr5_vgpr6_vgpr7 killed $exec
	v_mov_b32_e32 v5, v10
	v_mov_b32_e32 v6, v9
	v_mov_b32_e32 v7, v8
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[0:1], v[0:1], v[4:7], off sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e64 s[2:3], v[0:1], v[2:3]
	s_or_b64 s[0:1], s[2:3], s[0:1]
	s_nop 0
	v_writelane_b32 v22, s2, 18
	s_nop 1
	v_writelane_b32 v22, s3, 19
	s_mov_b64 s[2:3], s[0:1]
	v_writelane_b32 v22, s2, 20
	s_nop 1
	v_writelane_b32 v22, s3, 21
	scratch_store_dwordx2 off, v[0:1], s33 offset:120 ; 8-byte Folded Spill
	s_mov_b64 s[2:3], s[0:1]
	v_writelane_b32 v22, s2, 24
	s_nop 1
	v_writelane_b32 v22, s3, 25
	s_or_saveexec_b64 s[14:15], -1
	scratch_store_dword off, v22, s33       ; 4-byte Folded Spill
	s_mov_b64 exec, s[14:15]
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB3_10
; %bb.11:
	s_or_saveexec_b64 s[14:15], -1
	scratch_load_dword v22, off, s33        ; 4-byte Folded Reload
	s_mov_b64 exec, s[14:15]
	s_waitcnt vmcnt(0)
	v_readlane_b32 s0, v22, 24
	v_readlane_b32 s1, v22, 25
	s_or_b64 exec, exec, s[0:1]
; %bb.12:
	s_branch .LBB3_14
.LBB3_13:
	s_or_saveexec_b64 s[14:15], -1
	scratch_load_dword v22, off, s33        ; 4-byte Folded Reload
	s_mov_b64 exec, s[14:15]
	s_waitcnt vmcnt(0)
	v_readlane_b32 s0, v22, 16
	v_readlane_b32 s1, v22, 17
	s_or_b64 exec, exec, s[0:1]
	s_branch .LBB3_15
.LBB3_14:
	s_or_saveexec_b64 s[14:15], -1
	scratch_load_dword v22, off, s33        ; 4-byte Folded Reload
	s_mov_b64 exec, s[14:15]
	s_waitcnt vmcnt(0)
	v_readlane_b32 s0, v22, 22
	v_readlane_b32 s1, v22, 23
	s_or_b64 exec, exec, s[0:1]
	scratch_load_dwordx2 v[0:1], off, s33 offset:12 ; 8-byte Folded Reload
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[2:3], v[0:1], off offset:16
	s_mov_b32 s0, 32
	s_waitcnt vmcnt(0)
	v_lshrrev_b64 v[0:1], s0, v[2:3]
	v_mov_b32_e32 v1, v0
	v_mov_b32_e32 v0, v2
	s_getpc_b64 s[0:1]
	s_add_u32 s0, s0, __ockl_hsa_signal_add@rel32@lo+4
	s_addc_u32 s1, s1, __ockl_hsa_signal_add@rel32@hi+12
	v_mov_b32_e32 v2, 1
	v_mov_b32_e32 v3, 0
	v_mov_b32_e32 v4, 3
	s_swappc_b64 s[30:31], s[0:1]
	s_branch .LBB3_13
.LBB3_15:
	scratch_load_dwordx2 v[0:1], off, s33 offset:80 ; 8-byte Folded Reload
	s_mov_b64 s[0:1], 20
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[0:1], v[0:1], 0, s[0:1]
	scratch_store_dwordx2 off, v[0:1], s33 offset:144 ; 8-byte Folded Spill
.LBB3_16:                               ; =>This Inner Loop Header: Depth=1
	s_or_saveexec_b64 s[14:15], -1
	scratch_load_dword v22, off, s33        ; 4-byte Folded Reload
	s_mov_b64 exec, s[14:15]
	s_waitcnt vmcnt(0)
	v_readlane_b32 s2, v22, 0
	v_readlane_b32 s3, v22, 1
	v_mov_b32_e32 v0, 1
	scratch_store_dword off, v0, s33 offset:152 ; 4-byte Folded Spill
	s_mov_b64 s[0:1], exec
	v_writelane_b32 v22, s0, 26
	s_nop 1
	v_writelane_b32 v22, s1, 27
	s_or_saveexec_b64 s[14:15], -1
	scratch_store_dword off, v22, s33       ; 4-byte Folded Spill
	s_mov_b64 exec, s[14:15]
	s_and_b64 s[0:1], s[0:1], s[2:3]
	s_mov_b64 exec, s[0:1]
	s_cbranch_execz .LBB3_18
; %bb.17:                               ;   in Loop: Header=BB3_16 Depth=1
	scratch_load_dwordx2 v[0:1], off, s33 offset:144 ; 8-byte Folded Reload
	s_waitcnt vmcnt(0)
	global_load_dword v0, v[0:1], off sc0 sc1
	s_waitcnt vmcnt(0)
	buffer_inv sc0 sc1
	s_mov_b32 s0, 1
	v_and_b32_e64 v0, v0, s0
	scratch_store_dword off, v0, s33 offset:152 ; 4-byte Folded Spill
.LBB3_18:                               ;   in Loop: Header=BB3_16 Depth=1
	s_or_saveexec_b64 s[14:15], -1
	scratch_load_dword v22, off, s33        ; 4-byte Folded Reload
	s_mov_b64 exec, s[14:15]
	s_waitcnt vmcnt(0)
	v_readlane_b32 s0, v22, 26
	v_readlane_b32 s1, v22, 27
	s_or_b64 exec, exec, s[0:1]
	scratch_load_dword v0, off, s33 offset:152 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s2, v0
	s_mov_b64 s[0:1], -1
	s_mov_b32 s3, 0
	s_cmp_eq_u32 s2, s3
	v_writelane_b32 v22, s0, 28
	s_nop 1
	v_writelane_b32 v22, s1, 29
	s_mov_b64 s[14:15], exec
	s_mov_b64 exec, -1
	scratch_store_dword off, v22, s33       ; 4-byte Folded Spill
	s_mov_b64 exec, s[14:15]
	s_cbranch_scc1 .LBB3_20
; %bb.19:                               ;   in Loop: Header=BB3_16 Depth=1
	s_or_saveexec_b64 s[14:15], -1
	scratch_load_dword v22, off, s33        ; 4-byte Folded Reload
	s_mov_b64 exec, s[14:15]
	s_sleep 1
	s_mov_b64 s[0:1], 0
	s_waitcnt vmcnt(0)
	v_writelane_b32 v22, s0, 28
	s_nop 1
	v_writelane_b32 v22, s1, 29
	s_or_saveexec_b64 s[14:15], -1
	scratch_store_dword off, v22, s33       ; 4-byte Folded Spill
	s_mov_b64 exec, s[14:15]
.LBB3_20:                               ;   in Loop: Header=BB3_16 Depth=1
	s_or_saveexec_b64 s[14:15], -1
	scratch_load_dword v22, off, s33        ; 4-byte Folded Reload
	s_mov_b64 exec, s[14:15]
	s_waitcnt vmcnt(0)
	v_readlane_b32 s0, v22, 28
	v_readlane_b32 s1, v22, 29
	s_nop 1
	v_cndmask_b32_e64 v0, 0, 1, s[0:1]
	s_mov_b32 s0, 1
	v_cmp_ne_u32_e64 s[0:1], v0, s0
	s_and_b64 vcc, exec, s[0:1]
	s_cbranch_vccnz .LBB3_16
; %bb.21:
	s_or_saveexec_b64 s[14:15], -1
	scratch_load_dword v22, off, s33        ; 4-byte Folded Reload
	s_mov_b64 exec, s[14:15]
	s_waitcnt vmcnt(0)
	v_readlane_b32 s2, v22, 0
	v_readlane_b32 s3, v22, 1
	scratch_load_dwordx2 v[0:1], off, s33 offset:104 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[2:3], off, s33 offset:112 ; 8-byte Folded Reload
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[2:3], v[2:3], off
	s_waitcnt vmcnt(0)
	scratch_store_dwordx2 off, v[2:3], s33 offset:164 ; 8-byte Folded Spill
	global_load_dwordx2 v[0:1], v[0:1], off
	s_waitcnt vmcnt(0)
	scratch_store_dwordx2 off, v[0:1], s33 offset:156 ; 8-byte Folded Spill
	s_mov_b64 s[0:1], exec
	v_writelane_b32 v22, s0, 30
	s_nop 1
	v_writelane_b32 v22, s1, 31
	s_or_saveexec_b64 s[14:15], -1
	scratch_store_dword off, v22, s33       ; 4-byte Folded Spill
	s_mov_b64 exec, s[14:15]
	s_and_b64 s[0:1], s[0:1], s[2:3]
	s_mov_b64 exec, s[0:1]
	s_cbranch_execz .LBB3_27
; %bb.22:
	s_or_saveexec_b64 s[14:15], -1
	scratch_load_dword v22, off, s33        ; 4-byte Folded Reload
	s_mov_b64 exec, s[14:15]
	s_waitcnt vmcnt(0)
	v_readlane_b32 s0, v22, 10
	v_readlane_b32 s1, v22, 11
	scratch_load_dwordx2 v[0:1], off, s33 offset:12 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[2:3], off, s33 offset:88 ; 8-byte Folded Reload
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[12:13], v[2:3], off
	s_mov_b64 s[2:3], 1
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[2:3], v[12:13], 0, s[2:3]
	v_lshl_add_u64 v[4:5], v[2:3], 0, s[0:1]
	s_mov_b64 s[0:1], 0
	v_cmp_eq_u64_e64 s[2:3], v[4:5], s[0:1]
	v_mov_b32_e32 v7, v3
	v_mov_b32_e32 v6, v5
	v_cndmask_b32_e64 v10, v6, v7, s[2:3]
	v_mov_b32_e32 v3, v2
	v_mov_b32_e32 v2, v4
	v_cndmask_b32_e64 v4, v2, v3, s[2:3]
	v_mov_b32_e32 v8, v4
	v_mov_b32_e32 v9, v10
	v_mov_b64_e32 v[2:3], v[8:9]
	scratch_store_dwordx2 off, v[2:3], s33 offset:196 ; 8-byte Folded Spill
	s_mov_b64 s[2:3], 24
	v_lshl_add_u64 v[2:3], v[0:1], 0, s[2:3]
	scratch_store_dwordx2 off, v[2:3], s33 offset:188 ; 8-byte Folded Spill
	global_load_dwordx2 v[2:3], v[0:1], off offset:24 sc0 sc1
	s_nop 0
	global_load_dwordx2 v[6:7], v[0:1], off
	v_mov_b32_e32 v5, v9
	v_mov_b32_e32 v11, v13
	v_and_b32_e64 v5, v5, v11
                                        ; kill: def $vgpr8 killed $vgpr8 killed $vgpr8_vgpr9 killed $exec
	v_mov_b32_e32 v9, v12
	v_and_b32_e64 v14, v8, v9
                                        ; kill: def $vgpr14 killed $vgpr14 def $vgpr14_vgpr15 killed $exec
	v_mov_b32_e32 v15, v5
	v_mov_b32_e32 v5, v14
	s_mov_b32 s3, 24
	v_mad_u64_u32 v[12:13], s[4:5], v5, s3, 0
	v_mov_b32_e32 v8, v13
                                        ; implicit-def: $sgpr2
                                        ; implicit-def: $sgpr4
	v_mov_b32_e32 v5, s2
                                        ; kill: def $vgpr8 killed $vgpr8 def $vgpr8_vgpr9 killed $exec
	v_mov_b32_e32 v9, v5
	s_mov_b32 s2, 32
	v_lshrrev_b64 v[14:15], s2, v[14:15]
	v_mov_b32_e32 v5, v14
	v_mad_u64_u32 v[8:9], s[4:5], v5, s3, v[8:9]
                                        ; kill: def $vgpr8 killed $vgpr8 killed $vgpr8_vgpr9 killed $exec
                                        ; implicit-def: $sgpr3
                                        ; implicit-def: $sgpr4
	v_mov_b32_e32 v5, s3
                                        ; kill: def $vgpr8 killed $vgpr8 def $vgpr8_vgpr9 killed $exec
	v_mov_b32_e32 v9, v5
	v_lshlrev_b64 v[8:9], s2, v[8:9]
	v_mov_b32_e32 v11, v9
                                        ; kill: def $vgpr12 killed $vgpr12 killed $vgpr12_vgpr13 killed $exec
	s_mov_b32 s2, 0
	v_mov_b32_e32 v5, 0
                                        ; kill: def $vgpr12 killed $vgpr12 def $vgpr12_vgpr13 killed $exec
	v_mov_b32_e32 v13, v5
	v_mov_b32_e32 v5, v13
	v_or_b32_e64 v5, v5, v11
	v_mov_b32_e32 v9, v8
	v_mov_b32_e32 v8, v12
	v_or_b32_e64 v8, v8, v9
                                        ; kill: def $vgpr8 killed $vgpr8 def $vgpr8_vgpr9 killed $exec
	v_mov_b32_e32 v9, v5
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[6:7], v[6:7], 0, v[8:9]
	scratch_store_dwordx2 off, v[6:7], s33 offset:180 ; 8-byte Folded Spill
	global_store_dwordx2 v[6:7], v[2:3], off
	v_mov_b32_e32 v8, v3
	v_mov_b32_e32 v9, v2
                                        ; kill: def $vgpr4 killed $vgpr4 def $vgpr4_vgpr5_vgpr6_vgpr7 killed $exec
	v_mov_b32_e32 v5, v10
	v_mov_b32_e32 v6, v9
	v_mov_b32_e32 v7, v8
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[0:1], v[0:1], v[4:7], off offset:24 sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e64 s[4:5], v[0:1], v[2:3]
	v_cmp_ne_u64_e64 s[2:3], v[0:1], v[2:3]
	s_nop 0
	v_writelane_b32 v22, s4, 32
	s_nop 1
	v_writelane_b32 v22, s5, 33
	v_writelane_b32 v22, s0, 34
	s_nop 1
	v_writelane_b32 v22, s1, 35
	scratch_store_dwordx2 off, v[0:1], s33 offset:172 ; 8-byte Folded Spill
	s_mov_b64 s[0:1], exec
	v_writelane_b32 v22, s0, 36
	s_nop 1
	v_writelane_b32 v22, s1, 37
	s_or_saveexec_b64 s[14:15], -1
	scratch_store_dword off, v22, s33       ; 4-byte Folded Spill
	s_mov_b64 exec, s[14:15]
	s_and_b64 s[0:1], s[0:1], s[2:3]
	s_mov_b64 exec, s[0:1]
	s_cbranch_execz .LBB3_26
.LBB3_23:                               ; =>This Inner Loop Header: Depth=1
	s_or_saveexec_b64 s[14:15], -1
	scratch_load_dword v22, off, s33        ; 4-byte Folded Reload
	s_mov_b64 exec, s[14:15]
	s_waitcnt vmcnt(0)
	v_readlane_b32 s0, v22, 34
	v_readlane_b32 s1, v22, 35
	v_readlane_b32 s2, v22, 32
	v_readlane_b32 s3, v22, 33
	scratch_load_dwordx2 v[2:3], off, s33 offset:172 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[0:1], off, s33 offset:188 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[4:5], off, s33 offset:196 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[6:7], off, s33 offset:180 ; 8-byte Folded Reload
	s_sleep 1
	s_waitcnt vmcnt(0)
	global_store_dwordx2 v[6:7], v[2:3], off
	v_mov_b32_e32 v8, v3
	v_mov_b32_e32 v9, v2
	v_mov_b32_e32 v10, v5
                                        ; kill: def $vgpr4 killed $vgpr4 def $vgpr4_vgpr5_vgpr6_vgpr7 killed $exec
	v_mov_b32_e32 v5, v10
	v_mov_b32_e32 v6, v9
	v_mov_b32_e32 v7, v8
	buffer_wbl2 sc0 sc1
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[0:1], v[0:1], v[4:7], off sc0 sc1
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e64 s[2:3], v[0:1], v[2:3]
	s_or_b64 s[0:1], s[2:3], s[0:1]
	s_nop 0
	v_writelane_b32 v22, s2, 32
	s_nop 1
	v_writelane_b32 v22, s3, 33
	s_mov_b64 s[2:3], s[0:1]
	v_writelane_b32 v22, s2, 34
	s_nop 1
	v_writelane_b32 v22, s3, 35
	scratch_store_dwordx2 off, v[0:1], s33 offset:172 ; 8-byte Folded Spill
	s_mov_b64 s[2:3], s[0:1]
	v_writelane_b32 v22, s2, 38
	s_nop 1
	v_writelane_b32 v22, s3, 39
	s_or_saveexec_b64 s[14:15], -1
	scratch_store_dword off, v22, s33       ; 4-byte Folded Spill
	s_mov_b64 exec, s[14:15]
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB3_23
; %bb.24:
	s_or_saveexec_b64 s[14:15], -1
	scratch_load_dword v22, off, s33        ; 4-byte Folded Reload
	s_mov_b64 exec, s[14:15]
	s_waitcnt vmcnt(0)
	v_readlane_b32 s0, v22, 38
	v_readlane_b32 s1, v22, 39
	s_or_b64 exec, exec, s[0:1]
; %bb.25:
.LBB3_26:
	s_or_saveexec_b64 s[14:15], -1
	scratch_load_dword v22, off, s33        ; 4-byte Folded Reload
	s_mov_b64 exec, s[14:15]
	s_waitcnt vmcnt(0)
	v_readlane_b32 s0, v22, 36
	v_readlane_b32 s1, v22, 37
	s_or_b64 exec, exec, s[0:1]
.LBB3_27:
	s_or_saveexec_b64 s[14:15], -1
	scratch_load_dword v22, off, s33        ; 4-byte Folded Reload
	s_mov_b64 exec, s[14:15]
	s_waitcnt vmcnt(0)
	v_readlane_b32 s0, v22, 30
	v_readlane_b32 s1, v22, 31
	s_or_b64 exec, exec, s[0:1]
	scratch_load_dwordx2 v[4:5], off, s33 offset:156 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[2:3], off, s33 offset:164 ; 8-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_mov_b32_e32 v0, v2
	v_mov_b32_e32 v1, v3
	v_mov_b32_e32 v2, v4
	v_mov_b32_e32 v3, v5
	v_readlane_b32 s30, v21, 0
	v_readlane_b32 s31, v21, 1
	s_mov_b32 s32, s33
	s_xor_saveexec_b64 s[0:1], -1
	scratch_load_dword v21, off, s33 offset:204 ; 4-byte Folded Reload
	scratch_load_dword v22, off, s33 offset:208 ; 4-byte Folded Reload
	s_mov_b64 exec, s[0:1]
	s_mov_b32 s33, s13
	s_waitcnt vmcnt(0)
	s_setpc_b64 s[30:31]
.Lfunc_end3:
	.size	__ockl_hostcall_internal, .Lfunc_end3-__ockl_hostcall_internal
                                        ; -- End function
	.set .L__ockl_hostcall_internal.num_vgpr, max(23, .L__ockl_hsa_signal_add.num_vgpr)
	.set .L__ockl_hostcall_internal.num_agpr, max(32, .L__ockl_hsa_signal_add.num_agpr)
	.set .L__ockl_hostcall_internal.numbered_sgpr, max(34, .L__ockl_hsa_signal_add.numbered_sgpr)
	.set .L__ockl_hostcall_internal.num_named_barrier, max(0, .L__ockl_hsa_signal_add.num_named_barrier)
	.set .L__ockl_hostcall_internal.private_seg_size, 224+max(.L__ockl_hsa_signal_add.private_seg_size)
	.set .L__ockl_hostcall_internal.uses_vcc, or(1, .L__ockl_hsa_signal_add.uses_vcc)
	.set .L__ockl_hostcall_internal.uses_flat_scratch, or(0, .L__ockl_hsa_signal_add.uses_flat_scratch)
	.set .L__ockl_hostcall_internal.has_dyn_sized_stack, or(0, .L__ockl_hsa_signal_add.has_dyn_sized_stack)
	.set .L__ockl_hostcall_internal.has_recursion, or(0, .L__ockl_hsa_signal_add.has_recursion)
	.set .L__ockl_hostcall_internal.has_indirect_call, or(0, .L__ockl_hsa_signal_add.has_indirect_call)
	.section	.AMDGPU.csdata,"",@progbits
; Function info:
; codeLenInByte = 5084
; TotalNumSgprs: 40
; NumVgprs: 23
; NumAgprs: 32
; TotalNumVgprs: 56
; ScratchSize: 276
; MemoryBound: 0
	.text
	.p2align	2                               ; -- Begin function __ockl_hostcall_preview
	.type	__ockl_hostcall_preview,@function
__ockl_hostcall_preview:                ; @__ockl_hostcall_preview
; %bb.0:
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
	s_mov_b32 s18, s33
	s_mov_b32 s33, s32
	s_xor_saveexec_b64 s[0:1], -1
	scratch_store_dword off, v23, s33       ; 4-byte Folded Spill
	s_mov_b64 exec, s[0:1]
	s_add_i32 s32, s32, 8
	v_writelane_b32 v23, s30, 0
	s_nop 1
	v_writelane_b32 v23, s31, 1
	v_mov_b32_e32 v18, v16
	v_mov_b32_e32 v17, v15
	v_mov_b32_e32 v16, v14
	v_mov_b32_e32 v15, v13
	v_mov_b32_e32 v14, v12
	v_mov_b32_e32 v13, v11
	v_mov_b32_e32 v12, v10
	v_mov_b32_e32 v11, v9
	v_mov_b32_e32 v10, v8
	v_mov_b32_e32 v9, v7
	v_mov_b32_e32 v8, v6
	v_mov_b32_e32 v7, v5
	v_mov_b32_e32 v6, v4
	v_mov_b32_e32 v5, v3
	v_mov_b32_e32 v4, v2
	v_mov_b32_e32 v3, v1
	v_mov_b32_e32 v2, v0
	s_getpc_b64 s[0:1]
	s_add_u32 s0, s0, __oclc_ABI_version@rel32@lo+4
	s_addc_u32 s1, s1, __oclc_ABI_version@rel32@hi+12
	s_load_dword s0, s[0:1], 0x0
	s_mov_b32 s1, 0x1f4
	s_waitcnt lgkmcnt(0)
	s_cmp_lt_i32 s0, s1
	s_mov_b64 s[2:3], 0x50
	s_mov_b32 s1, s3
	s_mov_b64 s[16:17], 24
	s_mov_b32 s0, s17
	s_cselect_b32 s0, s0, s1
                                        ; kill: def $sgpr2 killed $sgpr2 killed $sgpr2_sgpr3
	s_mov_b32 s1, s16
	s_cselect_b32 s16, s1, s2
                                        ; kill: def $sgpr16 killed $sgpr16 def $sgpr16_sgpr17
	s_mov_b32 s17, s0
	s_mov_b32 s0, s8
	s_mov_b32 s1, s9
	s_mov_b32 s3, s16
	s_mov_b32 s2, s17
	s_add_u32 s0, s0, s3
	s_addc_u32 s2, s1, s2
                                        ; kill: def $sgpr0 killed $sgpr0 def $sgpr0_sgpr1
	s_mov_b32 s1, s2
	s_load_dwordx2 s[0:1], s[0:1], 0x0
	s_waitcnt lgkmcnt(0)
	s_mov_b32 s3, s0
	s_mov_b32 s2, 32
	s_lshr_b64 s[0:1], s[0:1], s2
	s_mov_b32 s2, s0
	s_getpc_b64 s[0:1]
	s_add_u32 s0, s0, __ockl_hostcall_internal@rel32@lo+4
	s_addc_u32 s1, s1, __ockl_hostcall_internal@rel32@hi+12
	v_mov_b32_e32 v0, s3
	v_mov_b32_e32 v1, s2
	v_readlane_b32 s30, v23, 0
	v_readlane_b32 s31, v23, 1
	s_mov_b32 s32, s33
	s_xor_saveexec_b64 s[2:3], -1
	scratch_load_dword v23, off, s33        ; 4-byte Folded Reload
	s_mov_b64 exec, s[2:3]
	s_mov_b32 s33, s18
	s_setpc_b64 s[0:1]
.Lfunc_end4:
	.size	__ockl_hostcall_preview, .Lfunc_end4-__ockl_hostcall_preview
                                        ; -- End function
	.set .L__ockl_hostcall_preview.num_vgpr, max(24, .L__ockl_hostcall_internal.num_vgpr)
	.set .L__ockl_hostcall_preview.num_agpr, max(0, .L__ockl_hostcall_internal.num_agpr)
	.set .L__ockl_hostcall_preview.numbered_sgpr, max(34, .L__ockl_hostcall_internal.numbered_sgpr)
	.set .L__ockl_hostcall_preview.num_named_barrier, max(0, .L__ockl_hostcall_internal.num_named_barrier)
	.set .L__ockl_hostcall_preview.private_seg_size, 8+max(.L__ockl_hostcall_internal.private_seg_size)
	.set .L__ockl_hostcall_preview.uses_vcc, or(1, .L__ockl_hostcall_internal.uses_vcc)
	.set .L__ockl_hostcall_preview.uses_flat_scratch, or(0, .L__ockl_hostcall_internal.uses_flat_scratch)
	.set .L__ockl_hostcall_preview.has_dyn_sized_stack, or(0, .L__ockl_hostcall_internal.has_dyn_sized_stack)
	.set .L__ockl_hostcall_preview.has_recursion, or(0, .L__ockl_hostcall_internal.has_recursion)
	.set .L__ockl_hostcall_preview.has_indirect_call, or(0, .L__ockl_hostcall_internal.has_indirect_call)
	.section	.AMDGPU.csdata,"",@progbits
; Function info:
; codeLenInByte = 328
; TotalNumSgprs: 40
; NumVgprs: 24
; NumAgprs: 32
; TotalNumVgprs: 56
; ScratchSize: 284
; MemoryBound: 0
	.text
	.p2align	2                               ; -- Begin function __ockl_fprintf_stderr_begin
	.type	__ockl_fprintf_stderr_begin,@function
__ockl_fprintf_stderr_begin:            ; @__ockl_fprintf_stderr_begin
; %bb.0:
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
	s_mov_b32 s19, s33
	s_mov_b32 s33, s32
	s_xor_saveexec_b64 s[0:1], -1
	scratch_store_dword off, v24, s33       ; 4-byte Folded Spill
	s_mov_b64 exec, s[0:1]
	s_add_i32 s32, s32, 16
	v_writelane_b32 v24, s30, 0
	s_nop 1
	v_writelane_b32 v24, s31, 1
	s_getpc_b64 s[0:1]
	s_add_u32 s0, s0, __ockl_hostcall_preview@rel32@lo+4
	s_addc_u32 s1, s1, __ockl_hostcall_preview@rel32@hi+12
	v_mov_b32_e32 v0, 2
	v_mov_b32_e32 v1, 33
	v_mov_b32_e32 v3, 1
	v_mov_b32_e32 v16, 0
	v_mov_b32_e32 v2, v16
	v_mov_b32_e32 v4, v16
	v_mov_b32_e32 v5, v16
	v_mov_b32_e32 v6, v16
	v_mov_b32_e32 v7, v16
	v_mov_b32_e32 v8, v16
	v_mov_b32_e32 v9, v16
	v_mov_b32_e32 v10, v16
	v_mov_b32_e32 v11, v16
	v_mov_b32_e32 v12, v16
	v_mov_b32_e32 v13, v16
	v_mov_b32_e32 v14, v16
	v_mov_b32_e32 v15, v16
	s_swappc_b64 s[30:31], s[0:1]
                                        ; implicit-def: $sgpr0
                                        ; implicit-def: $sgpr1
	v_mov_b32_e32 v2, s0
                                        ; kill: def $vgpr2 killed $vgpr2 def $vgpr2_vgpr3 killed $exec
	v_mov_b32_e32 v3, v1
	s_mov_b32 s0, 32
	v_lshrrev_b64 v[2:3], s0, v[2:3]
	v_mov_b32_e32 v1, v2
	v_readlane_b32 s30, v24, 0
	v_readlane_b32 s31, v24, 1
	s_mov_b32 s32, s33
	s_xor_saveexec_b64 s[0:1], -1
	scratch_load_dword v24, off, s33        ; 4-byte Folded Reload
	s_mov_b64 exec, s[0:1]
	s_mov_b32 s33, s19
	s_waitcnt vmcnt(0)
	s_setpc_b64 s[30:31]
.Lfunc_end5:
	.size	__ockl_fprintf_stderr_begin, .Lfunc_end5-__ockl_fprintf_stderr_begin
                                        ; -- End function
	.set .L__ockl_fprintf_stderr_begin.num_vgpr, max(25, .L__ockl_hostcall_preview.num_vgpr)
	.set .L__ockl_fprintf_stderr_begin.num_agpr, max(0, .L__ockl_hostcall_preview.num_agpr)
	.set .L__ockl_fprintf_stderr_begin.numbered_sgpr, max(34, .L__ockl_hostcall_preview.numbered_sgpr)
	.set .L__ockl_fprintf_stderr_begin.num_named_barrier, max(0, .L__ockl_hostcall_preview.num_named_barrier)
	.set .L__ockl_fprintf_stderr_begin.private_seg_size, 16+max(.L__ockl_hostcall_preview.private_seg_size)
	.set .L__ockl_fprintf_stderr_begin.uses_vcc, or(1, .L__ockl_hostcall_preview.uses_vcc)
	.set .L__ockl_fprintf_stderr_begin.uses_flat_scratch, or(0, .L__ockl_hostcall_preview.uses_flat_scratch)
	.set .L__ockl_fprintf_stderr_begin.has_dyn_sized_stack, or(0, .L__ockl_hostcall_preview.has_dyn_sized_stack)
	.set .L__ockl_fprintf_stderr_begin.has_recursion, or(0, .L__ockl_hostcall_preview.has_recursion)
	.set .L__ockl_fprintf_stderr_begin.has_indirect_call, or(0, .L__ockl_hostcall_preview.has_indirect_call)
	.section	.AMDGPU.csdata,"",@progbits
; Function info:
; codeLenInByte = 216
; TotalNumSgprs: 40
; NumVgprs: 25
; NumAgprs: 32
; TotalNumVgprs: 60
; ScratchSize: 300
; MemoryBound: 0
	.text
	.p2align	2                               ; -- Begin function __ockl_fprintf_append_string_n
	.type	__ockl_fprintf_append_string_n,@function
__ockl_fprintf_append_string_n:         ; @__ockl_fprintf_append_string_n
; %bb.0:
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
	s_mov_b32 s19, s33
	s_mov_b32 s33, s32
	s_xor_saveexec_b64 s[0:1], -1
	scratch_store_dword off, v30, s33 offset:656 ; 4-byte Folded Spill
	scratch_store_dword off, v34, s33 offset:660 ; 4-byte Folded Spill
	scratch_store_dword off, v35, s33 offset:664 ; 4-byte Folded Spill
	s_mov_b64 exec, s[0:1]
	s_add_i32 s32, s32, 0x2a0
	v_writelane_b32 v30, s30, 0
	s_nop 1
	v_writelane_b32 v30, s31, 1
	scratch_store_dword off, v31, s33 offset:44 ; 4-byte Folded Spill
	scratch_store_dword off, v6, s33 offset:40 ; 4-byte Folded Spill
	scratch_store_dword off, v5, s33 offset:36 ; 4-byte Folded Spill
	v_mov_b32_e32 v7, v3
	v_mov_b32_e32 v3, v2
	scratch_load_dword v2, off, s33 offset:40 ; 4-byte Folded Reload
	s_nop 0
	scratch_store_dword off, v3, s33 offset:32 ; 4-byte Folded Spill
	v_mov_b32_e32 v3, v1
	scratch_load_dword v1, off, s33 offset:36 ; 4-byte Folded Reload
	v_mov_b32_e32 v6, v0
	scratch_load_dword v0, off, s33 offset:32 ; 4-byte Folded Reload
                                        ; implicit-def: $vgpr35 : SGPR spill to VGPR lane
	v_writelane_b32 v35, s15, 0
	v_writelane_b32 v35, s14, 1
	v_writelane_b32 v35, s13, 2
	v_writelane_b32 v35, s12, 3
	v_writelane_b32 v35, s10, 4
	s_nop 1
	v_writelane_b32 v35, s11, 5
	v_writelane_b32 v35, s8, 6
	s_nop 1
	v_writelane_b32 v35, s9, 7
	v_writelane_b32 v35, s6, 8
	s_nop 1
	v_writelane_b32 v35, s7, 9
	v_writelane_b32 v35, s4, 10
	s_nop 1
	v_writelane_b32 v35, s5, 11
                                        ; kill: def $vgpr4 killed $vgpr4 def $vgpr4_vgpr5 killed $exec
	s_waitcnt vmcnt(1)
	v_mov_b32_e32 v5, v1
                                        ; kill: def $vgpr0 killed $vgpr0 def $vgpr0_vgpr1 killed $exec
	v_mov_b32_e32 v1, v7
                                        ; kill: def $vgpr6 killed $vgpr6 def $vgpr6_vgpr7 killed $exec
	v_mov_b32_e32 v7, v3
	scratch_store_dwordx2 off, v[4:5], s33 offset:24 ; 8-byte Folded Spill
	s_waitcnt vmcnt(1)
	v_mov_b64_e32 v[4:5], v[0:1]
	scratch_store_dwordx2 off, v[4:5], s33 offset:16 ; 8-byte Folded Spill
	s_mov_b32 s0, 0
	v_cmp_eq_u32_e64 s[0:1], v2, s0
	v_mov_b32_e32 v4, v7
	s_mov_b64 s[2:3], 2
	s_mov_b32 s4, s3
	v_or_b32_e64 v2, v4, s4
	v_mov_b32_e32 v3, v6
                                        ; kill: def $sgpr2 killed $sgpr2 killed $sgpr2_sgpr3
	v_or_b32_e64 v6, v3, s2
                                        ; kill: def $vgpr6 killed $vgpr6 def $vgpr6_vgpr7 killed $exec
	v_mov_b32_e32 v7, v2
	v_mov_b32_e32 v2, v7
	v_cndmask_b32_e64 v4, v2, v4, s[0:1]
	v_mov_b32_e32 v2, v6
	v_cndmask_b32_e64 v2, v2, v3, s[0:1]
                                        ; kill: def $vgpr2 killed $vgpr2 def $vgpr2_vgpr3 killed $exec
	v_mov_b32_e32 v3, v4
	scratch_store_dwordx2 off, v[2:3], s33 offset:8 ; 8-byte Folded Spill
	s_mov_b64 s[0:1], 0
	v_cmp_ne_u64_e64 s[0:1], v[0:1], s[0:1]
                                        ; implicit-def: $vgpr0_vgpr1_vgpr2_vgpr3
	s_mov_b64 s[2:3], exec
	s_and_b64 s[0:1], s[2:3], s[0:1]
	s_xor_b64 s[2:3], s[0:1], s[2:3]
	v_writelane_b32 v35, s2, 12
	s_nop 1
	v_writelane_b32 v35, s3, 13
	s_or_saveexec_b64 s[22:23], -1
	scratch_store_dword off, v35, s33       ; 4-byte Folded Spill
	s_mov_b64 exec, s[22:23]
	s_mov_b64 exec, s[0:1]
	s_cbranch_execz .LBB6_3
	s_branch .LBB6_2
.LBB6_1:
	s_or_saveexec_b64 s[22:23], -1
	scratch_load_dword v35, off, s33        ; 4-byte Folded Reload
	s_mov_b64 exec, s[22:23]
	s_waitcnt vmcnt(0)
	v_readlane_b32 s15, v35, 0
	v_readlane_b32 s14, v35, 1
	v_readlane_b32 s13, v35, 2
	v_readlane_b32 s12, v35, 3
	v_readlane_b32 s10, v35, 4
	v_readlane_b32 s11, v35, 5
	v_readlane_b32 s8, v35, 6
	v_readlane_b32 s9, v35, 7
	v_readlane_b32 s6, v35, 8
	v_readlane_b32 s7, v35, 9
	v_readlane_b32 s4, v35, 10
	v_readlane_b32 s5, v35, 11
	scratch_load_dword v31, off, s33 offset:44 ; 4-byte Folded Reload
	scratch_load_dwordx2 v[2:3], off, s33 offset:8 ; 8-byte Folded Reload
	s_mov_b32 s0, 0xffffff1f
	s_mov_b32 s1, -1
	s_mov_b32 s2, s1
	s_waitcnt vmcnt(0)
	v_mov_b32_e32 v0, v3
	v_and_b32_e64 v4, v0, s2
                                        ; kill: def $sgpr0 killed $sgpr0 killed $sgpr0_sgpr1
	v_mov_b32_e32 v0, v2
	v_and_b32_e64 v0, v0, s0
                                        ; kill: def $vgpr0 killed $vgpr0 def $vgpr0_vgpr1 killed $exec
	v_mov_b32_e32 v1, v4
	v_mov_b32_e32 v4, v1
	s_mov_b64 s[0:1], 32
	s_mov_b32 s2, s1
	v_or_b32_e64 v4, v4, s2
                                        ; kill: def $vgpr0 killed $vgpr0 killed $vgpr0_vgpr1 killed $exec
                                        ; kill: def $sgpr0 killed $sgpr0 killed $sgpr0_sgpr1
	v_or_b32_e64 v0, v0, s0
                                        ; kill: def $vgpr0 killed $vgpr0 def $vgpr0_vgpr1 killed $exec
	v_mov_b32_e32 v1, v4
	v_mov_b32_e32 v1, v0
	s_mov_b32 s0, 32
	v_lshrrev_b64 v[2:3], s0, v[2:3]
                                        ; kill: def $vgpr2 killed $vgpr2 killed $vgpr2_vgpr3 killed $exec
	s_getpc_b64 s[0:1]
	s_add_u32 s0, s0, __ockl_hostcall_preview@rel32@lo+4
	s_addc_u32 s1, s1, __ockl_hostcall_preview@rel32@hi+12
	v_mov_b32_e32 v0, 2
	v_mov_b32_e32 v16, 0
	scratch_store_dword off, v16, s33 offset:64 ; 4-byte Folded Spill
	v_mov_b32_e32 v3, v16
	v_mov_b32_e32 v4, v16
	v_mov_b32_e32 v5, v16
	v_mov_b32_e32 v6, v16
	v_mov_b32_e32 v7, v16
	v_mov_b32_e32 v8, v16
	v_mov_b32_e32 v9, v16
	v_mov_b32_e32 v10, v16
	v_mov_b32_e32 v11, v16
	v_mov_b32_e32 v12, v16
	v_mov_b32_e32 v13, v16
	v_mov_b32_e32 v14, v16
	v_mov_b32_e32 v15, v16
	s_swappc_b64 s[30:31], s[0:1]
	v_mov_b32_e32 v6, v1
	v_mov_b32_e32 v5, v2
	v_mov_b32_e32 v4, v3
                                        ; kill: def $vgpr0 killed $vgpr0 def $vgpr0_vgpr1_vgpr2_vgpr3 killed $exec
	v_mov_b32_e32 v1, v6
	v_mov_b32_e32 v2, v5
	v_mov_b32_e32 v3, v4
	scratch_store_dwordx4 off, v[0:3], s33 offset:48 ; 16-byte Folded Spill
	s_branch .LBB6_63
.LBB6_2:
	s_or_saveexec_b64 s[22:23], -1
	scratch_load_dword v35, off, s33        ; 4-byte Folded Reload
	s_mov_b64 exec, s[22:23]
	scratch_load_dwordx2 v[4:5], off, s33 offset:16 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[6:7], off, s33 offset:24 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[2:3], off, s33 offset:8 ; 8-byte Folded Reload
	s_mov_b64 s[0:1], 2
	s_mov_b32 s2, s1
	s_waitcnt vmcnt(0)
	v_mov_b32_e32 v1, v3
	v_and_b32_e64 v8, v1, s2
                                        ; kill: def $sgpr0 killed $sgpr0 killed $sgpr0_sgpr1
	v_mov_b32_e32 v0, v2
	v_and_b32_e64 v2, v0, s0
                                        ; kill: def $vgpr2 killed $vgpr2 def $vgpr2_vgpr3 killed $exec
	v_mov_b32_e32 v3, v8
	scratch_store_dwordx2 off, v[2:3], s33 offset:100 ; 8-byte Folded Spill
	s_mov_b64 s[0:1], -3
	s_mov_b32 s2, s1
	v_and_b32_e64 v2, v1, s2
                                        ; kill: def $sgpr0 killed $sgpr0 killed $sgpr0_sgpr1
	v_and_b32_e64 v0, v0, s0
                                        ; kill: def $vgpr0 killed $vgpr0 def $vgpr0_vgpr1 killed $exec
	v_mov_b32_e32 v1, v2
	v_mov_b32_e32 v10, v1
                                        ; kill: def $vgpr0 killed $vgpr0 killed $vgpr0_vgpr1 killed $exec
	s_mov_b64 s[0:1], 0
	s_mov_b32 s2, s1
	s_mov_b32 s3, s0
	v_mov_b32_e32 v9, s3
	v_mov_b32_e32 v8, s2
                                        ; kill: def $vgpr0 killed $vgpr0 def $vgpr0_vgpr1_vgpr2_vgpr3 killed $exec
	v_mov_b32_e32 v1, v10
	v_mov_b32_e32 v2, v9
	v_mov_b32_e32 v3, v8
	v_writelane_b32 v35, s0, 14
	s_nop 1
	v_writelane_b32 v35, s1, 15
	s_or_saveexec_b64 s[22:23], -1
	scratch_store_dword off, v35, s33       ; 4-byte Folded Spill
	s_mov_b64 exec, s[22:23]
	scratch_store_dwordx2 off, v[6:7], s33 offset:92 ; 8-byte Folded Spill
	scratch_store_dwordx2 off, v[4:5], s33 offset:84 ; 8-byte Folded Spill
	scratch_store_dwordx4 off, v[0:3], s33 offset:68 ; 16-byte Folded Spill
	s_branch .LBB6_4
.LBB6_3:
	s_or_saveexec_b64 s[22:23], -1
	scratch_load_dword v35, off, s33        ; 4-byte Folded Reload
	s_mov_b64 exec, s[22:23]
	s_waitcnt vmcnt(0)
	v_readlane_b32 s0, v35, 12
	v_readlane_b32 s1, v35, 13
	s_or_saveexec_b64 s[0:1], s[0:1]
	scratch_load_dwordx4 v[0:3], off, s33 offset:108 ; 16-byte Folded Reload
	s_waitcnt vmcnt(0)
	scratch_store_dwordx4 off, v[0:3], s33 offset:48 ; 16-byte Folded Spill
	s_and_b64 s[0:1], exec, s[0:1]
	v_writelane_b32 v35, s0, 16
	s_nop 1
	v_writelane_b32 v35, s1, 17
	s_or_saveexec_b64 s[22:23], -1
	scratch_store_dword off, v35, s33       ; 4-byte Folded Spill
	s_mov_b64 exec, s[22:23]
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execz .LBB6_63
	s_branch .LBB6_1
.LBB6_4:                                ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB6_8 Depth 2
                                        ;     Child Loop BB6_16 Depth 2
                                        ;     Child Loop BB6_24 Depth 2
                                        ;     Child Loop BB6_32 Depth 2
                                        ;     Child Loop BB6_40 Depth 2
                                        ;     Child Loop BB6_48 Depth 2
                                        ;     Child Loop BB6_56 Depth 2
	s_or_saveexec_b64 s[22:23], -1
	scratch_load_dword v35, off, s33        ; 4-byte Folded Reload
	s_mov_b64 exec, s[22:23]
	s_waitcnt vmcnt(0)
	v_readlane_b32 s0, v35, 14
	v_readlane_b32 s1, v35, 15
	scratch_load_dwordx2 v[4:5], off, s33 offset:100 ; 8-byte Folded Reload
	scratch_load_dwordx4 v[10:13], off, s33 offset:68 ; 16-byte Folded Reload
	scratch_load_dwordx2 v[2:3], off, s33 offset:92 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[0:1], off, s33 offset:84 ; 8-byte Folded Reload
	s_waitcnt vmcnt(0)
	scratch_store_dwordx2 off, v[0:1], s33 offset:168 ; 8-byte Folded Spill
	scratch_store_dwordx2 off, v[2:3], s33 offset:160 ; 8-byte Folded Spill
	v_writelane_b32 v35, s0, 18
	s_nop 1
	v_writelane_b32 v35, s1, 19
	s_mov_b64 s[4:5], 56
	v_cmp_gt_u64_e64 s[0:1], v[2:3], s[4:5]
	v_mov_b32_e32 v8, v11
	v_mov_b32_e32 v6, v10
                                        ; kill: def $vgpr6 killed $vgpr6 def $vgpr6_vgpr7 killed $exec
	v_mov_b32_e32 v7, v8
	scratch_store_dwordx2 off, v[6:7], s33 offset:152 ; 8-byte Folded Spill
	v_cmp_lt_u64_e64 s[2:3], v[2:3], s[4:5]
	v_mov_b32_e32 v7, v3
	s_mov_b32 s6, s5
	v_mov_b32_e32 v6, s6
	v_cndmask_b32_e64 v8, v6, v7, s[2:3]
	v_mov_b32_e32 v7, v2
                                        ; kill: def $sgpr4 killed $sgpr4 killed $sgpr4_sgpr5
	v_mov_b32_e32 v6, s4
	v_cndmask_b32_e64 v6, v6, v7, s[2:3]
	scratch_store_dword off, v6, s33 offset:148 ; 4-byte Folded Spill
                                        ; kill: def $vgpr6 killed $vgpr6 def $vgpr6_vgpr7 killed $exec
	v_mov_b32_e32 v7, v8
	scratch_store_dwordx2 off, v[6:7], s33 offset:140 ; 8-byte Folded Spill
	s_mov_b64 s[2:3], 0
	s_mov_b32 s4, s3
	v_mov_b32_e32 v6, v5
	v_mov_b32_e32 v7, s4
	v_cndmask_b32_e64 v6, v6, v7, s[0:1]
                                        ; kill: def $sgpr2 killed $sgpr2 killed $sgpr2_sgpr3
	v_mov_b32_e32 v5, s2
	v_cndmask_b32_e64 v4, v4, v5, s[0:1]
                                        ; kill: def $vgpr4 killed $vgpr4 def $vgpr4_vgpr5 killed $exec
	v_mov_b32_e32 v5, v6
	scratch_store_dwordx2 off, v[4:5], s33 offset:132 ; 8-byte Folded Spill
	s_mov_b64 s[2:3], 8
	v_cmp_lt_u64_e64 s[0:1], v[2:3], s[2:3]
	v_lshl_add_u64 v[0:1], v[0:1], 0, s[2:3]
                                        ; implicit-def: $vgpr2_vgpr3
	scratch_store_dwordx2 off, v[0:1], s33 offset:124 ; 8-byte Folded Spill
	s_mov_b64 s[2:3], exec
	s_and_b64 s[0:1], s[2:3], s[0:1]
	s_xor_b64 s[2:3], s[0:1], s[2:3]
	v_writelane_b32 v35, s2, 20
	s_nop 1
	v_writelane_b32 v35, s3, 21
	s_or_saveexec_b64 s[22:23], -1
	scratch_store_dword off, v35, s33       ; 4-byte Folded Spill
	s_mov_b64 exec, s[22:23]
	s_mov_b64 exec, s[0:1]
	s_cbranch_execz .LBB6_6
; %bb.5:                                ;   in Loop: Header=BB6_4 Depth=1
	s_or_saveexec_b64 s[22:23], -1
	scratch_load_dword v35, off, s33        ; 4-byte Folded Reload
	s_mov_b64 exec, s[22:23]
	scratch_load_dwordx2 v[0:1], off, s33 offset:160 ; 8-byte Folded Reload
	s_mov_b64 s[4:5], 0
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e64 s[2:3], v[0:1], s[4:5]
	s_mov_b32 s0, 0
	v_mov_b64_e32 v[2:3], 0
	v_mov_b64_e32 v[0:1], 0
	v_writelane_b32 v35, s4, 22
	s_nop 1
	v_writelane_b32 v35, s5, 23
	v_writelane_b32 v35, s0, 24
	scratch_store_dwordx2 off, v[2:3], s33 offset:184 ; 8-byte Folded Spill
	scratch_store_dwordx2 off, v[0:1], s33 offset:176 ; 8-byte Folded Spill
	s_mov_b64 s[0:1], exec
	v_writelane_b32 v35, s0, 25
	s_nop 1
	v_writelane_b32 v35, s1, 26
	s_or_saveexec_b64 s[22:23], -1
	scratch_store_dword off, v35, s33       ; 4-byte Folded Spill
	s_mov_b64 exec, s[22:23]
	s_and_b64 s[0:1], s[0:1], s[2:3]
	s_mov_b64 exec, s[0:1]
	s_cbranch_execz .LBB6_11
	s_branch .LBB6_8
.LBB6_6:                                ;   in Loop: Header=BB6_4 Depth=1
	s_or_saveexec_b64 s[22:23], -1
	scratch_load_dword v35, off, s33        ; 4-byte Folded Reload
	s_mov_b64 exec, s[22:23]
	s_waitcnt vmcnt(0)
	v_readlane_b32 s0, v35, 20
	v_readlane_b32 s1, v35, 21
	s_or_saveexec_b64 s[0:1], s[0:1]
	scratch_load_dwordx2 v[0:1], off, s33 offset:212 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[2:3], off, s33 offset:124 ; 8-byte Folded Reload
	s_waitcnt vmcnt(0)
	scratch_store_dwordx2 off, v[2:3], s33 offset:204 ; 8-byte Folded Spill
	v_mov_b32_e32 v2, 0
	scratch_store_dword off, v2, s33 offset:200 ; 4-byte Folded Spill
	scratch_store_dwordx2 off, v[0:1], s33 offset:192 ; 8-byte Folded Spill
	s_and_b64 s[0:1], exec, s[0:1]
	v_writelane_b32 v35, s0, 27
	s_nop 1
	v_writelane_b32 v35, s1, 28
	s_or_saveexec_b64 s[22:23], -1
	scratch_store_dword off, v35, s33       ; 4-byte Folded Spill
	s_mov_b64 exec, s[22:23]
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execz .LBB6_12
; %bb.7:                                ;   in Loop: Header=BB6_4 Depth=1
	scratch_load_dword v2, off, s33 offset:148 ; 4-byte Folded Reload
	scratch_load_dwordx2 v[0:1], off, s33 offset:168 ; 8-byte Folded Reload
	s_waitcnt vmcnt(0)
	flat_load_dwordx2 v[0:1], v[0:1]
	s_mov_b32 s0, -8
	v_add_u32_e64 v2, v2, s0
	scratch_store_dword off, v2, s33 offset:200 ; 4-byte Folded Spill
	s_waitcnt vmcnt(0) lgkmcnt(0)
	scratch_store_dwordx2 off, v[0:1], s33 offset:192 ; 8-byte Folded Spill
	s_branch .LBB6_12
.LBB6_8:                                ;   Parent Loop BB6_4 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_or_saveexec_b64 s[22:23], -1
	scratch_load_dword v35, off, s33        ; 4-byte Folded Reload
	s_mov_b64 exec, s[22:23]
	s_waitcnt vmcnt(0)
	v_readlane_b32 s0, v35, 24
	v_readlane_b32 s4, v35, 22
	v_readlane_b32 s5, v35, 23
	scratch_load_dwordx2 v[4:5], off, s33 offset:184 ; 8-byte Folded Reload
	scratch_load_dword v2, off, s33 offset:148 ; 4-byte Folded Reload
	scratch_load_dwordx2 v[0:1], off, s33 offset:168 ; 8-byte Folded Reload
	s_mov_b32 s1, 0
	s_mov_b32 s2, s0
	s_mov_b32 s3, s1
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[0:1], v[0:1], 0, s[2:3]
	flat_load_ubyte v0, v[0:1]
	s_mov_b32 s2, 0xffff
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_and_b32_e64 v0, s2, v0
                                        ; kill: def $vgpr0 killed $vgpr0 def $vgpr0_vgpr1 killed $exec
	v_mov_b32_e32 v1, s1
	s_mov_b32 s1, 3
	s_lshl_b32 s1, s0, s1
	v_lshlrev_b64 v[0:1], s1, v[0:1]
	v_mov_b32_e32 v3, v1
	v_mov_b32_e32 v6, v5
	v_or_b32_e64 v3, v3, v6
                                        ; kill: def $vgpr0 killed $vgpr0 killed $vgpr0_vgpr1 killed $exec
	v_mov_b32_e32 v1, v4
	v_or_b32_e64 v0, v0, v1
                                        ; kill: def $vgpr0 killed $vgpr0 def $vgpr0_vgpr1 killed $exec
	v_mov_b32_e32 v1, v3
	s_mov_b32 s1, 1
	s_add_i32 s2, s0, s1
	v_cmp_eq_u32_e64 s[0:1], s2, v2
	s_or_b64 s[0:1], s[0:1], s[4:5]
	s_mov_b64 s[4:5], s[0:1]
	v_writelane_b32 v35, s4, 22
	s_nop 1
	v_writelane_b32 v35, s5, 23
	v_writelane_b32 v35, s2, 24
	v_mov_b64_e32 v[2:3], v[0:1]
	scratch_store_dwordx2 off, v[2:3], s33 offset:184 ; 8-byte Folded Spill
	scratch_store_dwordx2 off, v[0:1], s33 offset:220 ; 8-byte Folded Spill
	s_mov_b64 s[2:3], s[0:1]
	v_writelane_b32 v35, s2, 29
	s_nop 1
	v_writelane_b32 v35, s3, 30
	s_or_saveexec_b64 s[22:23], -1
	scratch_store_dword off, v35, s33       ; 4-byte Folded Spill
	s_mov_b64 exec, s[22:23]
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB6_8
; %bb.9:                                ;   in Loop: Header=BB6_4 Depth=1
	s_or_saveexec_b64 s[22:23], -1
	scratch_load_dword v35, off, s33        ; 4-byte Folded Reload
	s_mov_b64 exec, s[22:23]
	s_waitcnt vmcnt(0)
	v_readlane_b32 s0, v35, 29
	v_readlane_b32 s1, v35, 30
	s_or_b64 exec, exec, s[0:1]
; %bb.10:                               ;   in Loop: Header=BB6_4 Depth=1
	scratch_load_dwordx2 v[0:1], off, s33 offset:220 ; 8-byte Folded Reload
	s_waitcnt vmcnt(0)
	scratch_store_dwordx2 off, v[0:1], s33 offset:176 ; 8-byte Folded Spill
.LBB6_11:                               ;   in Loop: Header=BB6_4 Depth=1
	s_or_saveexec_b64 s[22:23], -1
	scratch_load_dword v35, off, s33        ; 4-byte Folded Reload
	s_mov_b64 exec, s[22:23]
	s_waitcnt vmcnt(0)
	v_readlane_b32 s0, v35, 25
	v_readlane_b32 s1, v35, 26
	s_or_b64 exec, exec, s[0:1]
	scratch_load_dwordx2 v[0:1], off, s33 offset:168 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[2:3], off, s33 offset:176 ; 8-byte Folded Reload
	s_waitcnt vmcnt(0)
	scratch_store_dwordx2 off, v[2:3], s33 offset:212 ; 8-byte Folded Spill
	scratch_store_dwordx2 off, v[0:1], s33 offset:124 ; 8-byte Folded Spill
	s_branch .LBB6_6
.LBB6_12:                               ;   in Loop: Header=BB6_4 Depth=1
	s_or_saveexec_b64 s[22:23], -1
	scratch_load_dword v35, off, s33        ; 4-byte Folded Reload
	s_mov_b64 exec, s[22:23]
	s_waitcnt vmcnt(0)
	v_readlane_b32 s0, v35, 27
	v_readlane_b32 s1, v35, 28
	s_or_b64 exec, exec, s[0:1]
	scratch_load_dwordx2 v[0:1], off, s33 offset:204 ; 8-byte Folded Reload
	scratch_load_dword v2, off, s33 offset:200 ; 4-byte Folded Reload
	scratch_load_dwordx2 v[4:5], off, s33 offset:192 ; 8-byte Folded Reload
	s_waitcnt vmcnt(0)
	scratch_store_dwordx2 off, v[4:5], s33 offset:240 ; 8-byte Folded Spill
	scratch_store_dword off, v2, s33 offset:236 ; 4-byte Folded Spill
	s_mov_b32 s0, 8
	v_cmp_lt_u32_e64 s[0:1], v2, s0
	s_mov_b64 s[2:3], 8
	v_lshl_add_u64 v[0:1], v[0:1], 0, s[2:3]
                                        ; implicit-def: $vgpr2_vgpr3
	scratch_store_dwordx2 off, v[0:1], s33 offset:228 ; 8-byte Folded Spill
	s_mov_b64 s[2:3], exec
	s_and_b64 s[0:1], s[2:3], s[0:1]
	s_xor_b64 s[2:3], s[0:1], s[2:3]
	v_writelane_b32 v35, s2, 31
	s_nop 1
	v_writelane_b32 v35, s3, 32
	s_or_saveexec_b64 s[22:23], -1
	scratch_store_dword off, v35, s33       ; 4-byte Folded Spill
	s_mov_b64 exec, s[22:23]
	s_mov_b64 exec, s[0:1]
	s_cbranch_execz .LBB6_14
; %bb.13:                               ;   in Loop: Header=BB6_4 Depth=1
	s_or_saveexec_b64 s[22:23], -1
	scratch_load_dword v35, off, s33        ; 4-byte Folded Reload
	s_mov_b64 exec, s[22:23]
	scratch_load_dword v0, off, s33 offset:236 ; 4-byte Folded Reload
	s_mov_b32 s0, 0
	s_waitcnt vmcnt(0)
	v_cmp_ne_u32_e64 s[2:3], v0, s0
	s_mov_b64 s[4:5], 0
	v_mov_b64_e32 v[2:3], 0
	v_mov_b64_e32 v[0:1], 0
	v_writelane_b32 v35, s4, 33
	s_nop 1
	v_writelane_b32 v35, s5, 34
	v_writelane_b32 v35, s0, 35
	scratch_store_dwordx2 off, v[2:3], s33 offset:256 ; 8-byte Folded Spill
	scratch_store_dwordx2 off, v[0:1], s33 offset:248 ; 8-byte Folded Spill
	s_mov_b64 s[0:1], exec
	v_writelane_b32 v35, s0, 36
	s_nop 1
	v_writelane_b32 v35, s1, 37
	s_or_saveexec_b64 s[22:23], -1
	scratch_store_dword off, v35, s33       ; 4-byte Folded Spill
	s_mov_b64 exec, s[22:23]
	s_and_b64 s[0:1], s[0:1], s[2:3]
	s_mov_b64 exec, s[0:1]
	s_cbranch_execz .LBB6_19
	s_branch .LBB6_16
.LBB6_14:                               ;   in Loop: Header=BB6_4 Depth=1
	s_or_saveexec_b64 s[22:23], -1
	scratch_load_dword v35, off, s33        ; 4-byte Folded Reload
	s_mov_b64 exec, s[22:23]
	s_waitcnt vmcnt(0)
	v_readlane_b32 s0, v35, 31
	v_readlane_b32 s1, v35, 32
	s_or_saveexec_b64 s[0:1], s[0:1]
	scratch_load_dwordx2 v[0:1], off, s33 offset:284 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[2:3], off, s33 offset:228 ; 8-byte Folded Reload
	s_waitcnt vmcnt(0)
	scratch_store_dwordx2 off, v[2:3], s33 offset:276 ; 8-byte Folded Spill
	v_mov_b32_e32 v2, 0
	scratch_store_dword off, v2, s33 offset:272 ; 4-byte Folded Spill
	scratch_store_dwordx2 off, v[0:1], s33 offset:264 ; 8-byte Folded Spill
	s_and_b64 s[0:1], exec, s[0:1]
	v_writelane_b32 v35, s0, 38
	s_nop 1
	v_writelane_b32 v35, s1, 39
	s_or_saveexec_b64 s[22:23], -1
	scratch_store_dword off, v35, s33       ; 4-byte Folded Spill
	s_mov_b64 exec, s[22:23]
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execz .LBB6_20
; %bb.15:                               ;   in Loop: Header=BB6_4 Depth=1
	scratch_load_dword v2, off, s33 offset:236 ; 4-byte Folded Reload
	scratch_load_dwordx2 v[0:1], off, s33 offset:204 ; 8-byte Folded Reload
	s_waitcnt vmcnt(0)
	flat_load_dwordx2 v[0:1], v[0:1]
	s_mov_b32 s0, -8
	v_add_u32_e64 v2, v2, s0
	scratch_store_dword off, v2, s33 offset:272 ; 4-byte Folded Spill
	s_waitcnt vmcnt(0) lgkmcnt(0)
	scratch_store_dwordx2 off, v[0:1], s33 offset:264 ; 8-byte Folded Spill
	s_branch .LBB6_20
.LBB6_16:                               ;   Parent Loop BB6_4 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_or_saveexec_b64 s[22:23], -1
	scratch_load_dword v35, off, s33        ; 4-byte Folded Reload
	s_mov_b64 exec, s[22:23]
	s_waitcnt vmcnt(0)
	v_readlane_b32 s0, v35, 35
	v_readlane_b32 s4, v35, 33
	v_readlane_b32 s5, v35, 34
	scratch_load_dwordx2 v[4:5], off, s33 offset:256 ; 8-byte Folded Reload
	scratch_load_dword v2, off, s33 offset:236 ; 4-byte Folded Reload
	scratch_load_dwordx2 v[0:1], off, s33 offset:204 ; 8-byte Folded Reload
	s_mov_b32 s1, 0
	s_mov_b32 s2, s0
	s_mov_b32 s3, s1
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[0:1], v[0:1], 0, s[2:3]
	flat_load_ubyte v0, v[0:1]
	s_mov_b32 s2, 0xffff
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_and_b32_e64 v0, s2, v0
                                        ; kill: def $vgpr0 killed $vgpr0 def $vgpr0_vgpr1 killed $exec
	v_mov_b32_e32 v1, s1
	s_mov_b32 s1, 3
	s_lshl_b32 s1, s0, s1
	v_lshlrev_b64 v[0:1], s1, v[0:1]
	v_mov_b32_e32 v3, v1
	v_mov_b32_e32 v6, v5
	v_or_b32_e64 v3, v3, v6
                                        ; kill: def $vgpr0 killed $vgpr0 killed $vgpr0_vgpr1 killed $exec
	v_mov_b32_e32 v1, v4
	v_or_b32_e64 v0, v0, v1
                                        ; kill: def $vgpr0 killed $vgpr0 def $vgpr0_vgpr1 killed $exec
	v_mov_b32_e32 v1, v3
	s_mov_b32 s1, 1
	s_add_i32 s2, s0, s1
	v_cmp_eq_u32_e64 s[0:1], s2, v2
	s_or_b64 s[0:1], s[0:1], s[4:5]
	s_mov_b64 s[4:5], s[0:1]
	v_writelane_b32 v35, s4, 33
	s_nop 1
	v_writelane_b32 v35, s5, 34
	v_writelane_b32 v35, s2, 35
	v_mov_b64_e32 v[2:3], v[0:1]
	scratch_store_dwordx2 off, v[2:3], s33 offset:256 ; 8-byte Folded Spill
	scratch_store_dwordx2 off, v[0:1], s33 offset:292 ; 8-byte Folded Spill
	s_mov_b64 s[2:3], s[0:1]
	v_writelane_b32 v35, s2, 40
	s_nop 1
	v_writelane_b32 v35, s3, 41
	s_or_saveexec_b64 s[22:23], -1
	scratch_store_dword off, v35, s33       ; 4-byte Folded Spill
	s_mov_b64 exec, s[22:23]
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB6_16
; %bb.17:                               ;   in Loop: Header=BB6_4 Depth=1
	s_or_saveexec_b64 s[22:23], -1
	scratch_load_dword v35, off, s33        ; 4-byte Folded Reload
	s_mov_b64 exec, s[22:23]
	s_waitcnt vmcnt(0)
	v_readlane_b32 s0, v35, 40
	v_readlane_b32 s1, v35, 41
	s_or_b64 exec, exec, s[0:1]
; %bb.18:                               ;   in Loop: Header=BB6_4 Depth=1
	scratch_load_dwordx2 v[0:1], off, s33 offset:292 ; 8-byte Folded Reload
	s_waitcnt vmcnt(0)
	scratch_store_dwordx2 off, v[0:1], s33 offset:248 ; 8-byte Folded Spill
.LBB6_19:                               ;   in Loop: Header=BB6_4 Depth=1
	s_or_saveexec_b64 s[22:23], -1
	scratch_load_dword v35, off, s33        ; 4-byte Folded Reload
	s_mov_b64 exec, s[22:23]
	s_waitcnt vmcnt(0)
	v_readlane_b32 s0, v35, 36
	v_readlane_b32 s1, v35, 37
	s_or_b64 exec, exec, s[0:1]
	scratch_load_dwordx2 v[0:1], off, s33 offset:204 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[2:3], off, s33 offset:248 ; 8-byte Folded Reload
	s_waitcnt vmcnt(0)
	scratch_store_dwordx2 off, v[2:3], s33 offset:284 ; 8-byte Folded Spill
	scratch_store_dwordx2 off, v[0:1], s33 offset:228 ; 8-byte Folded Spill
	s_branch .LBB6_14
.LBB6_20:                               ;   in Loop: Header=BB6_4 Depth=1
	s_or_saveexec_b64 s[22:23], -1
	scratch_load_dword v35, off, s33        ; 4-byte Folded Reload
	s_mov_b64 exec, s[22:23]
	s_waitcnt vmcnt(0)
	v_readlane_b32 s0, v35, 38
	v_readlane_b32 s1, v35, 39
	s_or_b64 exec, exec, s[0:1]
	scratch_load_dwordx2 v[0:1], off, s33 offset:276 ; 8-byte Folded Reload
	scratch_load_dword v2, off, s33 offset:272 ; 4-byte Folded Reload
	scratch_load_dwordx2 v[4:5], off, s33 offset:264 ; 8-byte Folded Reload
	s_waitcnt vmcnt(0)
	scratch_store_dwordx2 off, v[4:5], s33 offset:312 ; 8-byte Folded Spill
	scratch_store_dword off, v2, s33 offset:308 ; 4-byte Folded Spill
	s_mov_b32 s0, 8
	v_cmp_lt_u32_e64 s[0:1], v2, s0
	s_mov_b64 s[2:3], 8
	v_lshl_add_u64 v[0:1], v[0:1], 0, s[2:3]
                                        ; implicit-def: $vgpr2_vgpr3
	scratch_store_dwordx2 off, v[0:1], s33 offset:300 ; 8-byte Folded Spill
	s_mov_b64 s[2:3], exec
	s_and_b64 s[0:1], s[2:3], s[0:1]
	s_xor_b64 s[2:3], s[0:1], s[2:3]
	v_writelane_b32 v35, s2, 42
	s_nop 1
	v_writelane_b32 v35, s3, 43
	s_or_saveexec_b64 s[22:23], -1
	scratch_store_dword off, v35, s33       ; 4-byte Folded Spill
	s_mov_b64 exec, s[22:23]
	s_mov_b64 exec, s[0:1]
	s_cbranch_execz .LBB6_22
; %bb.21:                               ;   in Loop: Header=BB6_4 Depth=1
	s_or_saveexec_b64 s[22:23], -1
	scratch_load_dword v35, off, s33        ; 4-byte Folded Reload
	s_mov_b64 exec, s[22:23]
	scratch_load_dword v0, off, s33 offset:308 ; 4-byte Folded Reload
	s_mov_b32 s0, 0
	s_waitcnt vmcnt(0)
	v_cmp_ne_u32_e64 s[2:3], v0, s0
	s_mov_b64 s[4:5], 0
	v_mov_b64_e32 v[2:3], 0
	v_mov_b64_e32 v[0:1], 0
	v_writelane_b32 v35, s4, 44
	s_nop 1
	v_writelane_b32 v35, s5, 45
	v_writelane_b32 v35, s0, 46
	scratch_store_dwordx2 off, v[2:3], s33 offset:328 ; 8-byte Folded Spill
	scratch_store_dwordx2 off, v[0:1], s33 offset:320 ; 8-byte Folded Spill
	s_mov_b64 s[0:1], exec
	v_writelane_b32 v35, s0, 47
	s_nop 1
	v_writelane_b32 v35, s1, 48
	s_or_saveexec_b64 s[22:23], -1
	scratch_store_dword off, v35, s33       ; 4-byte Folded Spill
	s_mov_b64 exec, s[22:23]
	s_and_b64 s[0:1], s[0:1], s[2:3]
	s_mov_b64 exec, s[0:1]
	s_cbranch_execz .LBB6_27
	s_branch .LBB6_24
.LBB6_22:                               ;   in Loop: Header=BB6_4 Depth=1
	s_or_saveexec_b64 s[22:23], -1
	scratch_load_dword v35, off, s33        ; 4-byte Folded Reload
	s_mov_b64 exec, s[22:23]
	s_waitcnt vmcnt(0)
	v_readlane_b32 s0, v35, 42
	v_readlane_b32 s1, v35, 43
	s_or_saveexec_b64 s[0:1], s[0:1]
	scratch_load_dwordx2 v[0:1], off, s33 offset:356 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[2:3], off, s33 offset:300 ; 8-byte Folded Reload
	s_waitcnt vmcnt(0)
	scratch_store_dwordx2 off, v[2:3], s33 offset:348 ; 8-byte Folded Spill
	v_mov_b32_e32 v2, 0
	scratch_store_dword off, v2, s33 offset:344 ; 4-byte Folded Spill
	scratch_store_dwordx2 off, v[0:1], s33 offset:336 ; 8-byte Folded Spill
	s_and_b64 s[0:1], exec, s[0:1]
	v_writelane_b32 v35, s0, 49
	s_nop 1
	v_writelane_b32 v35, s1, 50
	s_or_saveexec_b64 s[22:23], -1
	scratch_store_dword off, v35, s33       ; 4-byte Folded Spill
	s_mov_b64 exec, s[22:23]
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execz .LBB6_28
; %bb.23:                               ;   in Loop: Header=BB6_4 Depth=1
	scratch_load_dword v2, off, s33 offset:308 ; 4-byte Folded Reload
	scratch_load_dwordx2 v[0:1], off, s33 offset:276 ; 8-byte Folded Reload
	s_waitcnt vmcnt(0)
	flat_load_dwordx2 v[0:1], v[0:1]
	s_mov_b32 s0, -8
	v_add_u32_e64 v2, v2, s0
	scratch_store_dword off, v2, s33 offset:344 ; 4-byte Folded Spill
	s_waitcnt vmcnt(0) lgkmcnt(0)
	scratch_store_dwordx2 off, v[0:1], s33 offset:336 ; 8-byte Folded Spill
	s_branch .LBB6_28
.LBB6_24:                               ;   Parent Loop BB6_4 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_or_saveexec_b64 s[22:23], -1
	scratch_load_dword v35, off, s33        ; 4-byte Folded Reload
	s_mov_b64 exec, s[22:23]
	s_waitcnt vmcnt(0)
	v_readlane_b32 s0, v35, 46
	v_readlane_b32 s4, v35, 44
	v_readlane_b32 s5, v35, 45
	scratch_load_dwordx2 v[4:5], off, s33 offset:328 ; 8-byte Folded Reload
	scratch_load_dword v2, off, s33 offset:308 ; 4-byte Folded Reload
	scratch_load_dwordx2 v[0:1], off, s33 offset:276 ; 8-byte Folded Reload
	s_mov_b32 s1, 0
	s_mov_b32 s2, s0
	s_mov_b32 s3, s1
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[0:1], v[0:1], 0, s[2:3]
	flat_load_ubyte v0, v[0:1]
	s_mov_b32 s2, 0xffff
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_and_b32_e64 v0, s2, v0
                                        ; kill: def $vgpr0 killed $vgpr0 def $vgpr0_vgpr1 killed $exec
	v_mov_b32_e32 v1, s1
	s_mov_b32 s1, 3
	s_lshl_b32 s1, s0, s1
	v_lshlrev_b64 v[0:1], s1, v[0:1]
	v_mov_b32_e32 v3, v1
	v_mov_b32_e32 v6, v5
	v_or_b32_e64 v3, v3, v6
                                        ; kill: def $vgpr0 killed $vgpr0 killed $vgpr0_vgpr1 killed $exec
	v_mov_b32_e32 v1, v4
	v_or_b32_e64 v0, v0, v1
                                        ; kill: def $vgpr0 killed $vgpr0 def $vgpr0_vgpr1 killed $exec
	v_mov_b32_e32 v1, v3
	s_mov_b32 s1, 1
	s_add_i32 s2, s0, s1
	v_cmp_eq_u32_e64 s[0:1], s2, v2
	s_or_b64 s[0:1], s[0:1], s[4:5]
	s_mov_b64 s[4:5], s[0:1]
	v_writelane_b32 v35, s4, 44
	s_nop 1
	v_writelane_b32 v35, s5, 45
	v_writelane_b32 v35, s2, 46
	v_mov_b64_e32 v[2:3], v[0:1]
	scratch_store_dwordx2 off, v[2:3], s33 offset:328 ; 8-byte Folded Spill
	scratch_store_dwordx2 off, v[0:1], s33 offset:364 ; 8-byte Folded Spill
	s_mov_b64 s[2:3], s[0:1]
	v_writelane_b32 v35, s2, 51
	s_nop 1
	v_writelane_b32 v35, s3, 52
	s_or_saveexec_b64 s[22:23], -1
	scratch_store_dword off, v35, s33       ; 4-byte Folded Spill
	s_mov_b64 exec, s[22:23]
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB6_24
; %bb.25:                               ;   in Loop: Header=BB6_4 Depth=1
	s_or_saveexec_b64 s[22:23], -1
	scratch_load_dword v35, off, s33        ; 4-byte Folded Reload
	s_mov_b64 exec, s[22:23]
	s_waitcnt vmcnt(0)
	v_readlane_b32 s0, v35, 51
	v_readlane_b32 s1, v35, 52
	s_or_b64 exec, exec, s[0:1]
; %bb.26:                               ;   in Loop: Header=BB6_4 Depth=1
	scratch_load_dwordx2 v[0:1], off, s33 offset:364 ; 8-byte Folded Reload
	s_waitcnt vmcnt(0)
	scratch_store_dwordx2 off, v[0:1], s33 offset:320 ; 8-byte Folded Spill
.LBB6_27:                               ;   in Loop: Header=BB6_4 Depth=1
	s_or_saveexec_b64 s[22:23], -1
	scratch_load_dword v35, off, s33        ; 4-byte Folded Reload
	s_mov_b64 exec, s[22:23]
	s_waitcnt vmcnt(0)
	v_readlane_b32 s0, v35, 47
	v_readlane_b32 s1, v35, 48
	s_or_b64 exec, exec, s[0:1]
	scratch_load_dwordx2 v[0:1], off, s33 offset:276 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[2:3], off, s33 offset:320 ; 8-byte Folded Reload
	s_waitcnt vmcnt(0)
	scratch_store_dwordx2 off, v[2:3], s33 offset:356 ; 8-byte Folded Spill
	scratch_store_dwordx2 off, v[0:1], s33 offset:300 ; 8-byte Folded Spill
	s_branch .LBB6_22
.LBB6_28:                               ;   in Loop: Header=BB6_4 Depth=1
	s_or_saveexec_b64 s[22:23], -1
	scratch_load_dword v35, off, s33        ; 4-byte Folded Reload
	s_mov_b64 exec, s[22:23]
	s_waitcnt vmcnt(0)
	v_readlane_b32 s0, v35, 49
	v_readlane_b32 s1, v35, 50
	s_or_b64 exec, exec, s[0:1]
	scratch_load_dwordx2 v[0:1], off, s33 offset:348 ; 8-byte Folded Reload
	scratch_load_dword v2, off, s33 offset:344 ; 4-byte Folded Reload
	scratch_load_dwordx2 v[4:5], off, s33 offset:336 ; 8-byte Folded Reload
	s_waitcnt vmcnt(0)
	scratch_store_dwordx2 off, v[4:5], s33 offset:384 ; 8-byte Folded Spill
	scratch_store_dword off, v2, s33 offset:380 ; 4-byte Folded Spill
	s_mov_b32 s0, 8
	v_cmp_lt_u32_e64 s[0:1], v2, s0
	s_mov_b64 s[2:3], 8
	v_lshl_add_u64 v[0:1], v[0:1], 0, s[2:3]
                                        ; implicit-def: $vgpr2_vgpr3
	scratch_store_dwordx2 off, v[0:1], s33 offset:372 ; 8-byte Folded Spill
	s_mov_b64 s[2:3], exec
	s_and_b64 s[0:1], s[2:3], s[0:1]
	s_xor_b64 s[2:3], s[0:1], s[2:3]
	v_writelane_b32 v35, s2, 53
	s_nop 1
	v_writelane_b32 v35, s3, 54
	s_or_saveexec_b64 s[22:23], -1
	scratch_store_dword off, v35, s33       ; 4-byte Folded Spill
	s_mov_b64 exec, s[22:23]
	s_mov_b64 exec, s[0:1]
	s_cbranch_execz .LBB6_30
; %bb.29:                               ;   in Loop: Header=BB6_4 Depth=1
	s_or_saveexec_b64 s[22:23], -1
	scratch_load_dword v35, off, s33        ; 4-byte Folded Reload
	s_mov_b64 exec, s[22:23]
	scratch_load_dword v0, off, s33 offset:380 ; 4-byte Folded Reload
	s_mov_b32 s0, 0
	s_waitcnt vmcnt(0)
	v_cmp_ne_u32_e64 s[2:3], v0, s0
	s_mov_b64 s[4:5], 0
	v_mov_b64_e32 v[2:3], 0
	v_mov_b64_e32 v[0:1], 0
	v_writelane_b32 v35, s4, 55
	s_nop 1
	v_writelane_b32 v35, s5, 56
	v_writelane_b32 v35, s0, 57
	scratch_store_dwordx2 off, v[2:3], s33 offset:400 ; 8-byte Folded Spill
	scratch_store_dwordx2 off, v[0:1], s33 offset:392 ; 8-byte Folded Spill
	s_mov_b64 s[0:1], exec
	v_writelane_b32 v35, s0, 58
	s_nop 1
	v_writelane_b32 v35, s1, 59
	s_or_saveexec_b64 s[22:23], -1
	scratch_store_dword off, v35, s33       ; 4-byte Folded Spill
	s_mov_b64 exec, s[22:23]
	s_and_b64 s[0:1], s[0:1], s[2:3]
	s_mov_b64 exec, s[0:1]
	s_cbranch_execz .LBB6_35
	s_branch .LBB6_32
.LBB6_30:                               ;   in Loop: Header=BB6_4 Depth=1
	s_or_saveexec_b64 s[22:23], -1
	scratch_load_dword v35, off, s33        ; 4-byte Folded Reload
	s_mov_b64 exec, s[22:23]
	s_waitcnt vmcnt(0)
	v_readlane_b32 s0, v35, 53
	v_readlane_b32 s1, v35, 54
	s_or_saveexec_b64 s[0:1], s[0:1]
	scratch_load_dwordx2 v[0:1], off, s33 offset:428 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[2:3], off, s33 offset:372 ; 8-byte Folded Reload
	s_waitcnt vmcnt(0)
	scratch_store_dwordx2 off, v[2:3], s33 offset:420 ; 8-byte Folded Spill
	v_mov_b32_e32 v2, 0
	scratch_store_dword off, v2, s33 offset:416 ; 4-byte Folded Spill
	scratch_store_dwordx2 off, v[0:1], s33 offset:408 ; 8-byte Folded Spill
	s_and_b64 s[0:1], exec, s[0:1]
	v_writelane_b32 v35, s0, 60
	s_nop 1
	v_writelane_b32 v35, s1, 61
	s_or_saveexec_b64 s[22:23], -1
	scratch_store_dword off, v35, s33       ; 4-byte Folded Spill
	s_mov_b64 exec, s[22:23]
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execz .LBB6_36
; %bb.31:                               ;   in Loop: Header=BB6_4 Depth=1
	scratch_load_dword v2, off, s33 offset:380 ; 4-byte Folded Reload
	scratch_load_dwordx2 v[0:1], off, s33 offset:348 ; 8-byte Folded Reload
	s_waitcnt vmcnt(0)
	flat_load_dwordx2 v[0:1], v[0:1]
	s_mov_b32 s0, -8
	v_add_u32_e64 v2, v2, s0
	scratch_store_dword off, v2, s33 offset:416 ; 4-byte Folded Spill
	s_waitcnt vmcnt(0) lgkmcnt(0)
	scratch_store_dwordx2 off, v[0:1], s33 offset:408 ; 8-byte Folded Spill
	s_branch .LBB6_36
.LBB6_32:                               ;   Parent Loop BB6_4 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_or_saveexec_b64 s[22:23], -1
	scratch_load_dword v35, off, s33        ; 4-byte Folded Reload
	s_mov_b64 exec, s[22:23]
	s_waitcnt vmcnt(0)
	v_readlane_b32 s0, v35, 57
	v_readlane_b32 s4, v35, 55
	v_readlane_b32 s5, v35, 56
	scratch_load_dwordx2 v[4:5], off, s33 offset:400 ; 8-byte Folded Reload
	scratch_load_dword v2, off, s33 offset:380 ; 4-byte Folded Reload
	scratch_load_dwordx2 v[0:1], off, s33 offset:348 ; 8-byte Folded Reload
	s_mov_b32 s1, 0
	s_mov_b32 s2, s0
	s_mov_b32 s3, s1
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[0:1], v[0:1], 0, s[2:3]
	flat_load_ubyte v0, v[0:1]
	s_mov_b32 s2, 0xffff
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_and_b32_e64 v0, s2, v0
                                        ; kill: def $vgpr0 killed $vgpr0 def $vgpr0_vgpr1 killed $exec
	v_mov_b32_e32 v1, s1
	s_mov_b32 s1, 3
	s_lshl_b32 s1, s0, s1
	v_lshlrev_b64 v[0:1], s1, v[0:1]
	v_mov_b32_e32 v3, v1
	v_mov_b32_e32 v6, v5
	v_or_b32_e64 v3, v3, v6
                                        ; kill: def $vgpr0 killed $vgpr0 killed $vgpr0_vgpr1 killed $exec
	v_mov_b32_e32 v1, v4
	v_or_b32_e64 v0, v0, v1
                                        ; kill: def $vgpr0 killed $vgpr0 def $vgpr0_vgpr1 killed $exec
	v_mov_b32_e32 v1, v3
	s_mov_b32 s1, 1
	s_add_i32 s2, s0, s1
	v_cmp_eq_u32_e64 s[0:1], s2, v2
	s_or_b64 s[0:1], s[0:1], s[4:5]
	s_mov_b64 s[4:5], s[0:1]
	v_writelane_b32 v35, s4, 55
	s_nop 1
	v_writelane_b32 v35, s5, 56
	v_writelane_b32 v35, s2, 57
	v_mov_b64_e32 v[2:3], v[0:1]
	scratch_store_dwordx2 off, v[2:3], s33 offset:400 ; 8-byte Folded Spill
	scratch_store_dwordx2 off, v[0:1], s33 offset:436 ; 8-byte Folded Spill
	s_mov_b64 s[2:3], s[0:1]
	v_writelane_b32 v35, s2, 62
	s_nop 1
	v_writelane_b32 v35, s3, 63
	s_or_saveexec_b64 s[22:23], -1
	scratch_store_dword off, v35, s33       ; 4-byte Folded Spill
	s_mov_b64 exec, s[22:23]
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB6_32
; %bb.33:                               ;   in Loop: Header=BB6_4 Depth=1
	s_or_saveexec_b64 s[22:23], -1
	scratch_load_dword v35, off, s33        ; 4-byte Folded Reload
	s_mov_b64 exec, s[22:23]
	s_waitcnt vmcnt(0)
	v_readlane_b32 s0, v35, 62
	v_readlane_b32 s1, v35, 63
	s_or_b64 exec, exec, s[0:1]
; %bb.34:                               ;   in Loop: Header=BB6_4 Depth=1
	scratch_load_dwordx2 v[0:1], off, s33 offset:436 ; 8-byte Folded Reload
	s_waitcnt vmcnt(0)
	scratch_store_dwordx2 off, v[0:1], s33 offset:392 ; 8-byte Folded Spill
.LBB6_35:                               ;   in Loop: Header=BB6_4 Depth=1
	s_or_saveexec_b64 s[22:23], -1
	scratch_load_dword v35, off, s33        ; 4-byte Folded Reload
	s_mov_b64 exec, s[22:23]
	s_waitcnt vmcnt(0)
	v_readlane_b32 s0, v35, 58
	v_readlane_b32 s1, v35, 59
	s_or_b64 exec, exec, s[0:1]
	scratch_load_dwordx2 v[0:1], off, s33 offset:348 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[2:3], off, s33 offset:392 ; 8-byte Folded Reload
	s_waitcnt vmcnt(0)
	scratch_store_dwordx2 off, v[2:3], s33 offset:428 ; 8-byte Folded Spill
	scratch_store_dwordx2 off, v[0:1], s33 offset:372 ; 8-byte Folded Spill
	s_branch .LBB6_30
.LBB6_36:                               ;   in Loop: Header=BB6_4 Depth=1
	s_or_saveexec_b64 s[22:23], -1
	scratch_load_dword v35, off, s33        ; 4-byte Folded Reload
	s_mov_b64 exec, s[22:23]
	s_waitcnt vmcnt(0)
	v_readlane_b32 s0, v35, 60
	v_readlane_b32 s1, v35, 61
	s_or_b64 exec, exec, s[0:1]
	scratch_load_dwordx2 v[0:1], off, s33 offset:420 ; 8-byte Folded Reload
	scratch_load_dword v2, off, s33 offset:416 ; 4-byte Folded Reload
	scratch_load_dwordx2 v[4:5], off, s33 offset:408 ; 8-byte Folded Reload
	s_waitcnt vmcnt(0)
	scratch_store_dwordx2 off, v[4:5], s33 offset:456 ; 8-byte Folded Spill
	scratch_store_dword off, v2, s33 offset:452 ; 4-byte Folded Spill
	s_mov_b32 s0, 8
	v_cmp_lt_u32_e64 s[0:1], v2, s0
	s_mov_b64 s[2:3], 8
	v_lshl_add_u64 v[0:1], v[0:1], 0, s[2:3]
                                        ; implicit-def: $vgpr2_vgpr3
	scratch_store_dwordx2 off, v[0:1], s33 offset:444 ; 8-byte Folded Spill
	s_mov_b64 s[2:3], exec
	s_and_b64 s[0:1], s[2:3], s[0:1]
	s_xor_b64 s[2:3], s[0:1], s[2:3]
                                        ; implicit-def: $vgpr35 : SGPR spill to VGPR lane
	v_writelane_b32 v35, s2, 0
	s_nop 1
	v_writelane_b32 v35, s3, 1
	s_or_saveexec_b64 s[22:23], -1
	scratch_store_dword off, v35, s33 offset:4 ; 4-byte Folded Spill
	s_mov_b64 exec, s[22:23]
	s_mov_b64 exec, s[0:1]
	s_cbranch_execz .LBB6_38
; %bb.37:                               ;   in Loop: Header=BB6_4 Depth=1
	s_or_saveexec_b64 s[22:23], -1
	scratch_load_dword v35, off, s33 offset:4 ; 4-byte Folded Reload
	s_mov_b64 exec, s[22:23]
	scratch_load_dword v0, off, s33 offset:452 ; 4-byte Folded Reload
	s_mov_b32 s0, 0
	s_waitcnt vmcnt(0)
	v_cmp_ne_u32_e64 s[2:3], v0, s0
	s_mov_b64 s[4:5], 0
	v_mov_b64_e32 v[2:3], 0
	v_mov_b64_e32 v[0:1], 0
	v_writelane_b32 v35, s4, 2
	s_nop 1
	v_writelane_b32 v35, s5, 3
	v_writelane_b32 v35, s0, 4
	scratch_store_dwordx2 off, v[2:3], s33 offset:472 ; 8-byte Folded Spill
	scratch_store_dwordx2 off, v[0:1], s33 offset:464 ; 8-byte Folded Spill
	s_mov_b64 s[0:1], exec
	v_writelane_b32 v35, s0, 5
	s_nop 1
	v_writelane_b32 v35, s1, 6
	s_or_saveexec_b64 s[22:23], -1
	scratch_store_dword off, v35, s33 offset:4 ; 4-byte Folded Spill
	s_mov_b64 exec, s[22:23]
	s_and_b64 s[0:1], s[0:1], s[2:3]
	s_mov_b64 exec, s[0:1]
	s_cbranch_execz .LBB6_43
	s_branch .LBB6_40
.LBB6_38:                               ;   in Loop: Header=BB6_4 Depth=1
	s_or_saveexec_b64 s[22:23], -1
	scratch_load_dword v35, off, s33 offset:4 ; 4-byte Folded Reload
	s_mov_b64 exec, s[22:23]
	s_waitcnt vmcnt(0)
	v_readlane_b32 s0, v35, 0
	v_readlane_b32 s1, v35, 1
	s_or_saveexec_b64 s[0:1], s[0:1]
	scratch_load_dwordx2 v[0:1], off, s33 offset:500 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[2:3], off, s33 offset:444 ; 8-byte Folded Reload
	s_waitcnt vmcnt(0)
	scratch_store_dwordx2 off, v[2:3], s33 offset:492 ; 8-byte Folded Spill
	v_mov_b32_e32 v2, 0
	scratch_store_dword off, v2, s33 offset:488 ; 4-byte Folded Spill
	scratch_store_dwordx2 off, v[0:1], s33 offset:480 ; 8-byte Folded Spill
	s_and_b64 s[0:1], exec, s[0:1]
	v_writelane_b32 v35, s0, 7
	s_nop 1
	v_writelane_b32 v35, s1, 8
	s_or_saveexec_b64 s[22:23], -1
	scratch_store_dword off, v35, s33 offset:4 ; 4-byte Folded Spill
	s_mov_b64 exec, s[22:23]
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execz .LBB6_44
; %bb.39:                               ;   in Loop: Header=BB6_4 Depth=1
	scratch_load_dword v2, off, s33 offset:452 ; 4-byte Folded Reload
	scratch_load_dwordx2 v[0:1], off, s33 offset:420 ; 8-byte Folded Reload
	s_waitcnt vmcnt(0)
	flat_load_dwordx2 v[0:1], v[0:1]
	s_mov_b32 s0, -8
	v_add_u32_e64 v2, v2, s0
	scratch_store_dword off, v2, s33 offset:488 ; 4-byte Folded Spill
	s_waitcnt vmcnt(0) lgkmcnt(0)
	scratch_store_dwordx2 off, v[0:1], s33 offset:480 ; 8-byte Folded Spill
	s_branch .LBB6_44
.LBB6_40:                               ;   Parent Loop BB6_4 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_or_saveexec_b64 s[22:23], -1
	scratch_load_dword v35, off, s33 offset:4 ; 4-byte Folded Reload
	s_mov_b64 exec, s[22:23]
	s_waitcnt vmcnt(0)
	v_readlane_b32 s0, v35, 4
	v_readlane_b32 s4, v35, 2
	v_readlane_b32 s5, v35, 3
	scratch_load_dwordx2 v[4:5], off, s33 offset:472 ; 8-byte Folded Reload
	scratch_load_dword v2, off, s33 offset:452 ; 4-byte Folded Reload
	scratch_load_dwordx2 v[0:1], off, s33 offset:420 ; 8-byte Folded Reload
	s_mov_b32 s1, 0
	s_mov_b32 s2, s0
	s_mov_b32 s3, s1
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[0:1], v[0:1], 0, s[2:3]
	flat_load_ubyte v0, v[0:1]
	s_mov_b32 s2, 0xffff
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_and_b32_e64 v0, s2, v0
                                        ; kill: def $vgpr0 killed $vgpr0 def $vgpr0_vgpr1 killed $exec
	v_mov_b32_e32 v1, s1
	s_mov_b32 s1, 3
	s_lshl_b32 s1, s0, s1
	v_lshlrev_b64 v[0:1], s1, v[0:1]
	v_mov_b32_e32 v3, v1
	v_mov_b32_e32 v6, v5
	v_or_b32_e64 v3, v3, v6
                                        ; kill: def $vgpr0 killed $vgpr0 killed $vgpr0_vgpr1 killed $exec
	v_mov_b32_e32 v1, v4
	v_or_b32_e64 v0, v0, v1
                                        ; kill: def $vgpr0 killed $vgpr0 def $vgpr0_vgpr1 killed $exec
	v_mov_b32_e32 v1, v3
	s_mov_b32 s1, 1
	s_add_i32 s2, s0, s1
	v_cmp_eq_u32_e64 s[0:1], s2, v2
	s_or_b64 s[0:1], s[0:1], s[4:5]
	s_mov_b64 s[4:5], s[0:1]
	v_writelane_b32 v35, s4, 2
	s_nop 1
	v_writelane_b32 v35, s5, 3
	v_writelane_b32 v35, s2, 4
	v_mov_b64_e32 v[2:3], v[0:1]
	scratch_store_dwordx2 off, v[2:3], s33 offset:472 ; 8-byte Folded Spill
	scratch_store_dwordx2 off, v[0:1], s33 offset:508 ; 8-byte Folded Spill
	s_mov_b64 s[2:3], s[0:1]
	v_writelane_b32 v35, s2, 9
	s_nop 1
	v_writelane_b32 v35, s3, 10
	s_or_saveexec_b64 s[22:23], -1
	scratch_store_dword off, v35, s33 offset:4 ; 4-byte Folded Spill
	s_mov_b64 exec, s[22:23]
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB6_40
; %bb.41:                               ;   in Loop: Header=BB6_4 Depth=1
	s_or_saveexec_b64 s[22:23], -1
	scratch_load_dword v35, off, s33 offset:4 ; 4-byte Folded Reload
	s_mov_b64 exec, s[22:23]
	s_waitcnt vmcnt(0)
	v_readlane_b32 s0, v35, 9
	v_readlane_b32 s1, v35, 10
	s_or_b64 exec, exec, s[0:1]
; %bb.42:                               ;   in Loop: Header=BB6_4 Depth=1
	scratch_load_dwordx2 v[0:1], off, s33 offset:508 ; 8-byte Folded Reload
	s_waitcnt vmcnt(0)
	scratch_store_dwordx2 off, v[0:1], s33 offset:464 ; 8-byte Folded Spill
.LBB6_43:                               ;   in Loop: Header=BB6_4 Depth=1
	s_or_saveexec_b64 s[22:23], -1
	scratch_load_dword v35, off, s33 offset:4 ; 4-byte Folded Reload
	s_mov_b64 exec, s[22:23]
	s_waitcnt vmcnt(0)
	v_readlane_b32 s0, v35, 5
	v_readlane_b32 s1, v35, 6
	s_or_b64 exec, exec, s[0:1]
	scratch_load_dwordx2 v[0:1], off, s33 offset:420 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[2:3], off, s33 offset:464 ; 8-byte Folded Reload
	s_waitcnt vmcnt(0)
	scratch_store_dwordx2 off, v[2:3], s33 offset:500 ; 8-byte Folded Spill
	scratch_store_dwordx2 off, v[0:1], s33 offset:444 ; 8-byte Folded Spill
	s_branch .LBB6_38
.LBB6_44:                               ;   in Loop: Header=BB6_4 Depth=1
	s_or_saveexec_b64 s[22:23], -1
	scratch_load_dword v35, off, s33 offset:4 ; 4-byte Folded Reload
	s_mov_b64 exec, s[22:23]
	s_waitcnt vmcnt(0)
	v_readlane_b32 s0, v35, 7
	v_readlane_b32 s1, v35, 8
	s_or_b64 exec, exec, s[0:1]
	scratch_load_dwordx2 v[0:1], off, s33 offset:492 ; 8-byte Folded Reload
	scratch_load_dword v2, off, s33 offset:488 ; 4-byte Folded Reload
	scratch_load_dwordx2 v[4:5], off, s33 offset:480 ; 8-byte Folded Reload
	s_waitcnt vmcnt(0)
	scratch_store_dwordx2 off, v[4:5], s33 offset:528 ; 8-byte Folded Spill
	scratch_store_dword off, v2, s33 offset:524 ; 4-byte Folded Spill
	s_mov_b32 s0, 8
	v_cmp_lt_u32_e64 s[0:1], v2, s0
	s_mov_b64 s[2:3], 8
	v_lshl_add_u64 v[0:1], v[0:1], 0, s[2:3]
                                        ; implicit-def: $vgpr2_vgpr3
	scratch_store_dwordx2 off, v[0:1], s33 offset:516 ; 8-byte Folded Spill
	s_mov_b64 s[2:3], exec
	s_and_b64 s[0:1], s[2:3], s[0:1]
	s_xor_b64 s[2:3], s[0:1], s[2:3]
	v_writelane_b32 v35, s2, 11
	s_nop 1
	v_writelane_b32 v35, s3, 12
	s_or_saveexec_b64 s[22:23], -1
	scratch_store_dword off, v35, s33 offset:4 ; 4-byte Folded Spill
	s_mov_b64 exec, s[22:23]
	s_mov_b64 exec, s[0:1]
	s_cbranch_execz .LBB6_46
; %bb.45:                               ;   in Loop: Header=BB6_4 Depth=1
	s_or_saveexec_b64 s[22:23], -1
	scratch_load_dword v35, off, s33 offset:4 ; 4-byte Folded Reload
	s_mov_b64 exec, s[22:23]
	scratch_load_dword v0, off, s33 offset:524 ; 4-byte Folded Reload
	s_mov_b32 s0, 0
	s_waitcnt vmcnt(0)
	v_cmp_ne_u32_e64 s[2:3], v0, s0
	s_mov_b64 s[4:5], 0
	v_mov_b64_e32 v[2:3], 0
	v_mov_b64_e32 v[0:1], 0
	v_writelane_b32 v35, s4, 13
	s_nop 1
	v_writelane_b32 v35, s5, 14
	v_writelane_b32 v35, s0, 15
	scratch_store_dwordx2 off, v[2:3], s33 offset:544 ; 8-byte Folded Spill
	scratch_store_dwordx2 off, v[0:1], s33 offset:536 ; 8-byte Folded Spill
	s_mov_b64 s[0:1], exec
	v_writelane_b32 v35, s0, 16
	s_nop 1
	v_writelane_b32 v35, s1, 17
	s_or_saveexec_b64 s[22:23], -1
	scratch_store_dword off, v35, s33 offset:4 ; 4-byte Folded Spill
	s_mov_b64 exec, s[22:23]
	s_and_b64 s[0:1], s[0:1], s[2:3]
	s_mov_b64 exec, s[0:1]
	s_cbranch_execz .LBB6_51
	s_branch .LBB6_48
.LBB6_46:                               ;   in Loop: Header=BB6_4 Depth=1
	s_or_saveexec_b64 s[22:23], -1
	scratch_load_dword v35, off, s33 offset:4 ; 4-byte Folded Reload
	s_mov_b64 exec, s[22:23]
	s_waitcnt vmcnt(0)
	v_readlane_b32 s0, v35, 11
	v_readlane_b32 s1, v35, 12
	s_or_saveexec_b64 s[0:1], s[0:1]
	scratch_load_dwordx2 v[0:1], off, s33 offset:572 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[2:3], off, s33 offset:516 ; 8-byte Folded Reload
	s_waitcnt vmcnt(0)
	scratch_store_dwordx2 off, v[2:3], s33 offset:564 ; 8-byte Folded Spill
	v_mov_b32_e32 v2, 0
	scratch_store_dword off, v2, s33 offset:560 ; 4-byte Folded Spill
	scratch_store_dwordx2 off, v[0:1], s33 offset:552 ; 8-byte Folded Spill
	s_and_b64 s[0:1], exec, s[0:1]
	v_writelane_b32 v35, s0, 18
	s_nop 1
	v_writelane_b32 v35, s1, 19
	s_or_saveexec_b64 s[22:23], -1
	scratch_store_dword off, v35, s33 offset:4 ; 4-byte Folded Spill
	s_mov_b64 exec, s[22:23]
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execz .LBB6_52
; %bb.47:                               ;   in Loop: Header=BB6_4 Depth=1
	scratch_load_dword v2, off, s33 offset:524 ; 4-byte Folded Reload
	scratch_load_dwordx2 v[0:1], off, s33 offset:492 ; 8-byte Folded Reload
	s_waitcnt vmcnt(0)
	flat_load_dwordx2 v[0:1], v[0:1]
	s_mov_b32 s0, -8
	v_add_u32_e64 v2, v2, s0
	scratch_store_dword off, v2, s33 offset:560 ; 4-byte Folded Spill
	s_waitcnt vmcnt(0) lgkmcnt(0)
	scratch_store_dwordx2 off, v[0:1], s33 offset:552 ; 8-byte Folded Spill
	s_branch .LBB6_52
.LBB6_48:                               ;   Parent Loop BB6_4 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_or_saveexec_b64 s[22:23], -1
	scratch_load_dword v35, off, s33 offset:4 ; 4-byte Folded Reload
	s_mov_b64 exec, s[22:23]
	s_waitcnt vmcnt(0)
	v_readlane_b32 s0, v35, 15
	v_readlane_b32 s4, v35, 13
	v_readlane_b32 s5, v35, 14
	scratch_load_dwordx2 v[4:5], off, s33 offset:544 ; 8-byte Folded Reload
	scratch_load_dword v2, off, s33 offset:524 ; 4-byte Folded Reload
	scratch_load_dwordx2 v[0:1], off, s33 offset:492 ; 8-byte Folded Reload
	s_mov_b32 s1, 0
	s_mov_b32 s2, s0
	s_mov_b32 s3, s1
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[0:1], v[0:1], 0, s[2:3]
	flat_load_ubyte v0, v[0:1]
	s_mov_b32 s2, 0xffff
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_and_b32_e64 v0, s2, v0
                                        ; kill: def $vgpr0 killed $vgpr0 def $vgpr0_vgpr1 killed $exec
	v_mov_b32_e32 v1, s1
	s_mov_b32 s1, 3
	s_lshl_b32 s1, s0, s1
	v_lshlrev_b64 v[0:1], s1, v[0:1]
	v_mov_b32_e32 v3, v1
	v_mov_b32_e32 v6, v5
	v_or_b32_e64 v3, v3, v6
                                        ; kill: def $vgpr0 killed $vgpr0 killed $vgpr0_vgpr1 killed $exec
	v_mov_b32_e32 v1, v4
	v_or_b32_e64 v0, v0, v1
                                        ; kill: def $vgpr0 killed $vgpr0 def $vgpr0_vgpr1 killed $exec
	v_mov_b32_e32 v1, v3
	s_mov_b32 s1, 1
	s_add_i32 s2, s0, s1
	v_cmp_eq_u32_e64 s[0:1], s2, v2
	s_or_b64 s[0:1], s[0:1], s[4:5]
	s_mov_b64 s[4:5], s[0:1]
	v_writelane_b32 v35, s4, 13
	s_nop 1
	v_writelane_b32 v35, s5, 14
	v_writelane_b32 v35, s2, 15
	v_mov_b64_e32 v[2:3], v[0:1]
	scratch_store_dwordx2 off, v[2:3], s33 offset:544 ; 8-byte Folded Spill
	scratch_store_dwordx2 off, v[0:1], s33 offset:580 ; 8-byte Folded Spill
	s_mov_b64 s[2:3], s[0:1]
	v_writelane_b32 v35, s2, 20
	s_nop 1
	v_writelane_b32 v35, s3, 21
	s_or_saveexec_b64 s[22:23], -1
	scratch_store_dword off, v35, s33 offset:4 ; 4-byte Folded Spill
	s_mov_b64 exec, s[22:23]
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB6_48
; %bb.49:                               ;   in Loop: Header=BB6_4 Depth=1
	s_or_saveexec_b64 s[22:23], -1
	scratch_load_dword v35, off, s33 offset:4 ; 4-byte Folded Reload
	s_mov_b64 exec, s[22:23]
	s_waitcnt vmcnt(0)
	v_readlane_b32 s0, v35, 20
	v_readlane_b32 s1, v35, 21
	s_or_b64 exec, exec, s[0:1]
; %bb.50:                               ;   in Loop: Header=BB6_4 Depth=1
	scratch_load_dwordx2 v[0:1], off, s33 offset:580 ; 8-byte Folded Reload
	s_waitcnt vmcnt(0)
	scratch_store_dwordx2 off, v[0:1], s33 offset:536 ; 8-byte Folded Spill
.LBB6_51:                               ;   in Loop: Header=BB6_4 Depth=1
	s_or_saveexec_b64 s[22:23], -1
	scratch_load_dword v35, off, s33 offset:4 ; 4-byte Folded Reload
	s_mov_b64 exec, s[22:23]
	s_waitcnt vmcnt(0)
	v_readlane_b32 s0, v35, 16
	v_readlane_b32 s1, v35, 17
	s_or_b64 exec, exec, s[0:1]
	scratch_load_dwordx2 v[0:1], off, s33 offset:492 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[2:3], off, s33 offset:536 ; 8-byte Folded Reload
	s_waitcnt vmcnt(0)
	scratch_store_dwordx2 off, v[2:3], s33 offset:572 ; 8-byte Folded Spill
	scratch_store_dwordx2 off, v[0:1], s33 offset:516 ; 8-byte Folded Spill
	s_branch .LBB6_46
.LBB6_52:                               ;   in Loop: Header=BB6_4 Depth=1
	s_or_saveexec_b64 s[22:23], -1
	scratch_load_dword v35, off, s33 offset:4 ; 4-byte Folded Reload
	s_mov_b64 exec, s[22:23]
	s_waitcnt vmcnt(0)
	v_readlane_b32 s0, v35, 18
	v_readlane_b32 s1, v35, 19
	s_or_b64 exec, exec, s[0:1]
	scratch_load_dword v0, off, s33 offset:560 ; 4-byte Folded Reload
	scratch_load_dwordx2 v[2:3], off, s33 offset:552 ; 8-byte Folded Reload
	s_waitcnt vmcnt(0)
	scratch_store_dwordx2 off, v[2:3], s33 offset:592 ; 8-byte Folded Spill
	scratch_store_dword off, v0, s33 offset:588 ; 4-byte Folded Spill
	s_mov_b32 s0, 8
	v_cmp_lt_u32_e64 s[0:1], v0, s0
                                        ; implicit-def: $vgpr0_vgpr1
	s_mov_b64 s[2:3], exec
	s_and_b64 s[0:1], s[2:3], s[0:1]
	s_xor_b64 s[2:3], s[0:1], s[2:3]
	v_writelane_b32 v35, s2, 22
	s_nop 1
	v_writelane_b32 v35, s3, 23
	s_or_saveexec_b64 s[22:23], -1
	scratch_store_dword off, v35, s33 offset:4 ; 4-byte Folded Spill
	s_mov_b64 exec, s[22:23]
	s_mov_b64 exec, s[0:1]
	s_cbranch_execz .LBB6_54
; %bb.53:                               ;   in Loop: Header=BB6_4 Depth=1
	s_or_saveexec_b64 s[22:23], -1
	scratch_load_dword v35, off, s33 offset:4 ; 4-byte Folded Reload
	s_mov_b64 exec, s[22:23]
	scratch_load_dword v0, off, s33 offset:588 ; 4-byte Folded Reload
	s_mov_b32 s0, 0
	s_waitcnt vmcnt(0)
	v_cmp_ne_u32_e64 s[2:3], v0, s0
	s_mov_b64 s[4:5], 0
	v_mov_b64_e32 v[2:3], 0
	v_mov_b64_e32 v[0:1], 0
	v_writelane_b32 v35, s4, 24
	s_nop 1
	v_writelane_b32 v35, s5, 25
	v_writelane_b32 v35, s0, 26
	scratch_store_dwordx2 off, v[2:3], s33 offset:608 ; 8-byte Folded Spill
	scratch_store_dwordx2 off, v[0:1], s33 offset:600 ; 8-byte Folded Spill
	s_mov_b64 s[0:1], exec
	v_writelane_b32 v35, s0, 27
	s_nop 1
	v_writelane_b32 v35, s1, 28
	s_or_saveexec_b64 s[22:23], -1
	scratch_store_dword off, v35, s33 offset:4 ; 4-byte Folded Spill
	s_mov_b64 exec, s[22:23]
	s_and_b64 s[0:1], s[0:1], s[2:3]
	s_mov_b64 exec, s[0:1]
	s_cbranch_execz .LBB6_59
	s_branch .LBB6_56
.LBB6_54:                               ;   in Loop: Header=BB6_4 Depth=1
	s_or_saveexec_b64 s[22:23], -1
	scratch_load_dword v35, off, s33 offset:4 ; 4-byte Folded Reload
	s_mov_b64 exec, s[22:23]
	s_waitcnt vmcnt(0)
	v_readlane_b32 s0, v35, 22
	v_readlane_b32 s1, v35, 23
	s_or_saveexec_b64 s[0:1], s[0:1]
	scratch_load_dwordx2 v[0:1], off, s33 offset:624 ; 8-byte Folded Reload
	s_waitcnt vmcnt(0)
	scratch_store_dwordx2 off, v[0:1], s33 offset:616 ; 8-byte Folded Spill
	s_and_b64 s[0:1], exec, s[0:1]
	v_writelane_b32 v35, s0, 29
	s_nop 1
	v_writelane_b32 v35, s1, 30
	s_or_saveexec_b64 s[22:23], -1
	scratch_store_dword off, v35, s33 offset:4 ; 4-byte Folded Spill
	s_mov_b64 exec, s[22:23]
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execz .LBB6_60
; %bb.55:                               ;   in Loop: Header=BB6_4 Depth=1
	scratch_load_dwordx2 v[0:1], off, s33 offset:564 ; 8-byte Folded Reload
	s_waitcnt vmcnt(0)
	flat_load_dwordx2 v[0:1], v[0:1]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	scratch_store_dwordx2 off, v[0:1], s33 offset:616 ; 8-byte Folded Spill
	s_branch .LBB6_60
.LBB6_56:                               ;   Parent Loop BB6_4 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_or_saveexec_b64 s[22:23], -1
	scratch_load_dword v35, off, s33 offset:4 ; 4-byte Folded Reload
	s_mov_b64 exec, s[22:23]
	s_waitcnt vmcnt(0)
	v_readlane_b32 s0, v35, 26
	v_readlane_b32 s4, v35, 24
	v_readlane_b32 s5, v35, 25
	scratch_load_dwordx2 v[4:5], off, s33 offset:608 ; 8-byte Folded Reload
	scratch_load_dword v2, off, s33 offset:588 ; 4-byte Folded Reload
	scratch_load_dwordx2 v[0:1], off, s33 offset:564 ; 8-byte Folded Reload
	s_mov_b32 s1, 0
	s_mov_b32 s2, s0
	s_mov_b32 s3, s1
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[0:1], v[0:1], 0, s[2:3]
	flat_load_ubyte v0, v[0:1]
	s_mov_b32 s2, 0xffff
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_and_b32_e64 v0, s2, v0
                                        ; kill: def $vgpr0 killed $vgpr0 def $vgpr0_vgpr1 killed $exec
	v_mov_b32_e32 v1, s1
	s_mov_b32 s1, 3
	s_lshl_b32 s1, s0, s1
	v_lshlrev_b64 v[0:1], s1, v[0:1]
	v_mov_b32_e32 v3, v1
	v_mov_b32_e32 v6, v5
	v_or_b32_e64 v3, v3, v6
                                        ; kill: def $vgpr0 killed $vgpr0 killed $vgpr0_vgpr1 killed $exec
	v_mov_b32_e32 v1, v4
	v_or_b32_e64 v0, v0, v1
                                        ; kill: def $vgpr0 killed $vgpr0 def $vgpr0_vgpr1 killed $exec
	v_mov_b32_e32 v1, v3
	s_mov_b32 s1, 1
	s_add_i32 s2, s0, s1
	v_cmp_eq_u32_e64 s[0:1], s2, v2
	s_or_b64 s[0:1], s[0:1], s[4:5]
	s_mov_b64 s[4:5], s[0:1]
	v_writelane_b32 v35, s4, 24
	s_nop 1
	v_writelane_b32 v35, s5, 25
	v_writelane_b32 v35, s2, 26
	v_mov_b64_e32 v[2:3], v[0:1]
	scratch_store_dwordx2 off, v[2:3], s33 offset:608 ; 8-byte Folded Spill
	scratch_store_dwordx2 off, v[0:1], s33 offset:632 ; 8-byte Folded Spill
	s_mov_b64 s[2:3], s[0:1]
	v_writelane_b32 v35, s2, 31
	s_nop 1
	v_writelane_b32 v35, s3, 32
	s_or_saveexec_b64 s[22:23], -1
	scratch_store_dword off, v35, s33 offset:4 ; 4-byte Folded Spill
	s_mov_b64 exec, s[22:23]
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB6_56
; %bb.57:                               ;   in Loop: Header=BB6_4 Depth=1
	s_or_saveexec_b64 s[22:23], -1
	scratch_load_dword v35, off, s33 offset:4 ; 4-byte Folded Reload
	s_mov_b64 exec, s[22:23]
	s_waitcnt vmcnt(0)
	v_readlane_b32 s0, v35, 31
	v_readlane_b32 s1, v35, 32
	s_or_b64 exec, exec, s[0:1]
; %bb.58:                               ;   in Loop: Header=BB6_4 Depth=1
	scratch_load_dwordx2 v[0:1], off, s33 offset:632 ; 8-byte Folded Reload
	s_waitcnt vmcnt(0)
	scratch_store_dwordx2 off, v[0:1], s33 offset:600 ; 8-byte Folded Spill
.LBB6_59:                               ;   in Loop: Header=BB6_4 Depth=1
	s_or_saveexec_b64 s[22:23], -1
	scratch_load_dword v35, off, s33 offset:4 ; 4-byte Folded Reload
	s_mov_b64 exec, s[22:23]
	s_waitcnt vmcnt(0)
	v_readlane_b32 s0, v35, 27
	v_readlane_b32 s1, v35, 28
	s_or_b64 exec, exec, s[0:1]
	scratch_load_dwordx2 v[0:1], off, s33 offset:600 ; 8-byte Folded Reload
	s_waitcnt vmcnt(0)
	scratch_store_dwordx2 off, v[0:1], s33 offset:624 ; 8-byte Folded Spill
	s_branch .LBB6_54
.LBB6_60:                               ;   in Loop: Header=BB6_4 Depth=1
	s_or_saveexec_b64 s[22:23], -1
	scratch_load_dword v34, off, s33        ; 4-byte Folded Reload
	s_mov_b64 exec, s[22:23]
	s_or_saveexec_b64 s[22:23], -1
	scratch_load_dword v35, off, s33 offset:4 ; 4-byte Folded Reload
	s_mov_b64 exec, s[22:23]
	s_waitcnt vmcnt(0)
	v_readlane_b32 s0, v35, 29
	v_readlane_b32 s1, v35, 30
	s_or_b64 exec, exec, s[0:1]
	v_readlane_b32 s15, v34, 0
	v_readlane_b32 s14, v34, 1
	v_readlane_b32 s13, v34, 2
	v_readlane_b32 s12, v34, 3
	v_readlane_b32 s10, v34, 4
	v_readlane_b32 s11, v34, 5
	v_readlane_b32 s8, v34, 6
	v_readlane_b32 s9, v34, 7
	v_readlane_b32 s6, v34, 8
	v_readlane_b32 s7, v34, 9
	v_readlane_b32 s4, v34, 10
	v_readlane_b32 s5, v34, 11
	scratch_load_dwordx2 v[0:1], off, s33 offset:140 ; 8-byte Folded Reload
	scratch_load_dword v31, off, s33 offset:44 ; 4-byte Folded Reload
	scratch_load_dwordx2 v[20:21], off, s33 offset:592 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[22:23], off, s33 offset:528 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[24:25], off, s33 offset:456 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[26:27], off, s33 offset:384 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[28:29], off, s33 offset:312 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[32:33], off, s33 offset:240 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[4:5], off, s33 offset:132 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[2:3], off, s33 offset:152 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[18:19], off, s33 offset:616 ; 8-byte Folded Reload
	s_waitcnt vmcnt(10)
	v_mov_b32_e32 v1, v0
	s_mov_b32 s0, 28
	v_mov_b32_e32 v0, 2
	v_lshl_add_u32 v1, v1, v0, s0
	s_mov_b32 s0, 0x1e0
	v_and_b32_e64 v6, v1, s0
	s_mov_b32 s0, 0
	v_mov_b32_e32 v1, 0
                                        ; kill: def $vgpr6 killed $vgpr6 def $vgpr6_vgpr7 killed $exec
	v_mov_b32_e32 v7, v1
	s_mov_b32 s0, 0xffffff1f
	s_mov_b32 s1, -1
	s_mov_b32 s2, s1
	s_waitcnt vmcnt(1)
	v_mov_b32_e32 v1, v3
	v_and_b32_e64 v1, v1, s2
                                        ; kill: def $sgpr0 killed $sgpr0 killed $sgpr0_sgpr1
	v_and_b32_e64 v2, v2, s0
                                        ; kill: def $vgpr2 killed $vgpr2 def $vgpr2_vgpr3 killed $exec
	v_mov_b32_e32 v3, v1
	v_mov_b32_e32 v1, v3
	v_mov_b32_e32 v8, v5
	v_or_b32_e64 v1, v1, v8
                                        ; kill: def $vgpr2 killed $vgpr2 killed $vgpr2_vgpr3 killed $exec
	v_mov_b32_e32 v3, v4
	v_or_b32_e64 v2, v2, v3
                                        ; kill: def $vgpr2 killed $vgpr2 def $vgpr2_vgpr3 killed $exec
	v_mov_b32_e32 v3, v1
	v_mov_b32_e32 v1, v3
	v_mov_b32_e32 v4, v7
	v_or_b32_e64 v1, v1, v4
	v_mov_b32_e32 v4, v2
	v_mov_b32_e32 v5, v6
	v_or_b32_e64 v4, v4, v5
                                        ; kill: def $vgpr4 killed $vgpr4 def $vgpr4_vgpr5 killed $exec
	v_mov_b32_e32 v5, v1
	v_mov_b32_e32 v1, v4
	s_mov_b32 s0, 32
	v_writelane_b32 v35, s0, 33
	v_lshrrev_b64 v[2:3], s0, v[2:3]
                                        ; kill: def $vgpr2 killed $vgpr2 killed $vgpr2_vgpr3 killed $exec
	v_lshrrev_b64 v[4:5], s0, v[32:33]
                                        ; kill: def $vgpr4 killed $vgpr4 killed $vgpr4_vgpr5 killed $exec
	v_lshrrev_b64 v[6:7], s0, v[28:29]
                                        ; kill: def $vgpr6 killed $vgpr6 killed $vgpr6_vgpr7 killed $exec
	v_lshrrev_b64 v[8:9], s0, v[26:27]
                                        ; kill: def $vgpr8 killed $vgpr8 killed $vgpr8_vgpr9 killed $exec
	v_lshrrev_b64 v[10:11], s0, v[24:25]
                                        ; kill: def $vgpr10 killed $vgpr10 killed $vgpr10_vgpr11 killed $exec
	v_lshrrev_b64 v[12:13], s0, v[22:23]
                                        ; kill: def $vgpr12 killed $vgpr12 killed $vgpr12_vgpr13 killed $exec
	v_lshrrev_b64 v[14:15], s0, v[20:21]
                                        ; kill: def $vgpr14 killed $vgpr14 killed $vgpr14_vgpr15 killed $exec
	s_waitcnt vmcnt(0)
	v_lshrrev_b64 v[16:17], s0, v[18:19]
                                        ; kill: def $vgpr16 killed $vgpr16 killed $vgpr16_vgpr17 killed $exec
	v_mov_b32_e32 v3, v32
	v_mov_b32_e32 v5, v28
	v_mov_b32_e32 v7, v26
	v_mov_b32_e32 v9, v24
	v_mov_b32_e32 v11, v22
	v_mov_b32_e32 v13, v20
	v_mov_b32_e32 v15, v18
	s_getpc_b64 s[0:1]
	s_add_u32 s0, s0, __ockl_hostcall_preview@rel32@lo+4
	s_addc_u32 s1, s1, __ockl_hostcall_preview@rel32@hi+12
	s_swappc_b64 s[30:31], s[0:1]
	scratch_load_dwordx2 v[12:13], off, s33 offset:160 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[8:9], off, s33 offset:140 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[4:5], off, s33 offset:168 ; 8-byte Folded Reload
	v_readlane_b32 s2, v34, 18
	v_readlane_b32 s3, v34, 19
	v_mov_b32_e32 v10, v1
	v_mov_b32_e32 v7, v2
	v_mov_b32_e32 v6, v3
                                        ; kill: def $vgpr0 killed $vgpr0 def $vgpr0_vgpr1_vgpr2_vgpr3 killed $exec
	v_mov_b32_e32 v1, v10
	v_mov_b32_e32 v2, v7
	v_mov_b32_e32 v3, v6
	s_waitcnt vmcnt(2)
	v_mov_b32_e32 v6, v12
	s_waitcnt vmcnt(1)
	v_mov_b32_e32 v11, v8
	v_mov_b32_e32 v7, v13
	v_mov_b32_e32 v10, v9
	v_sub_co_u32_e64 v6, s[0:1], v6, v11
	s_nop 1
	v_subb_co_u32_e64 v10, s[0:1], v7, v10, s[0:1]
                                        ; kill: def $vgpr6 killed $vgpr6 def $vgpr6_vgpr7 killed $exec
	v_mov_b32_e32 v7, v10
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[4:5], v[4:5], 0, v[8:9]
	s_mov_b64 s[0:1], 0
	v_cmp_eq_u64_e64 s[0:1], v[6:7], s[0:1]
	s_or_b64 s[0:1], s[0:1], s[2:3]
	s_mov_b64 s[2:3], s[0:1]
	v_writelane_b32 v34, s2, 14
	s_nop 1
	v_writelane_b32 v34, s3, 15
	s_or_saveexec_b64 s[22:23], -1
	scratch_store_dword off, v34, s33       ; 4-byte Folded Spill
	s_mov_b64 exec, s[22:23]
	scratch_store_dwordx2 off, v[6:7], s33 offset:92 ; 8-byte Folded Spill
	scratch_store_dwordx2 off, v[4:5], s33 offset:84 ; 8-byte Folded Spill
	v_mov_b64_e32 v[6:7], v[2:3]
	v_mov_b64_e32 v[4:5], v[0:1]
	scratch_store_dwordx4 off, v[4:7], s33 offset:68 ; 16-byte Folded Spill
	scratch_store_dwordx4 off, v[0:3], s33 offset:640 ; 16-byte Folded Spill
	s_mov_b64 s[2:3], s[0:1]
	v_writelane_b32 v35, s2, 34
	s_nop 1
	v_writelane_b32 v35, s3, 35
	s_or_saveexec_b64 s[22:23], -1
	scratch_store_dword off, v35, s33 offset:4 ; 4-byte Folded Spill
	s_mov_b64 exec, s[22:23]
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB6_4
; %bb.61:
	s_or_saveexec_b64 s[22:23], -1
	scratch_load_dword v35, off, s33 offset:4 ; 4-byte Folded Reload
	s_mov_b64 exec, s[22:23]
	s_waitcnt vmcnt(0)
	v_readlane_b32 s0, v35, 34
	v_readlane_b32 s1, v35, 35
	s_or_b64 exec, exec, s[0:1]
; %bb.62:
	scratch_load_dwordx4 v[0:3], off, s33 offset:640 ; 16-byte Folded Reload
	s_waitcnt vmcnt(0)
	scratch_store_dwordx4 off, v[0:3], s33 offset:108 ; 16-byte Folded Spill
	s_branch .LBB6_3
.LBB6_63:
	s_or_saveexec_b64 s[22:23], -1
	scratch_load_dword v35, off, s33        ; 4-byte Folded Reload
	s_mov_b64 exec, s[22:23]
	s_waitcnt vmcnt(0)
	v_readlane_b32 s0, v35, 16
	v_readlane_b32 s1, v35, 17
	s_or_b64 exec, exec, s[0:1]
	scratch_load_dwordx4 v[4:7], off, s33 offset:48 ; 16-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_mov_b32_e32 v0, v5
                                        ; implicit-def: $sgpr0
                                        ; implicit-def: $sgpr1
	v_mov_b32_e32 v2, s0
                                        ; kill: def $vgpr2 killed $vgpr2 def $vgpr2_vgpr3 killed $exec
	v_mov_b32_e32 v3, v0
	v_mov_b32_e32 v0, v4
	s_mov_b32 s0, 32
	v_lshrrev_b64 v[2:3], s0, v[2:3]
	v_mov_b32_e32 v1, v2
	v_readlane_b32 s30, v30, 0
	v_readlane_b32 s31, v30, 1
	s_mov_b32 s32, s33
	s_xor_saveexec_b64 s[0:1], -1
	scratch_load_dword v30, off, s33 offset:656 ; 4-byte Folded Reload
	scratch_load_dword v34, off, s33 offset:660 ; 4-byte Folded Reload
	scratch_load_dword v35, off, s33 offset:664 ; 4-byte Folded Reload
	s_mov_b64 exec, s[0:1]
	s_mov_b32 s33, s19
	s_waitcnt vmcnt(0)
	s_setpc_b64 s[30:31]
.Lfunc_end6:
	.size	__ockl_fprintf_append_string_n, .Lfunc_end6-__ockl_fprintf_append_string_n
                                        ; -- End function
	.set .L__ockl_fprintf_append_string_n.num_vgpr, max(36, .L__ockl_hostcall_preview.num_vgpr)
	.set .L__ockl_fprintf_append_string_n.num_agpr, max(0, .L__ockl_hostcall_preview.num_agpr)
	.set .L__ockl_fprintf_append_string_n.numbered_sgpr, max(34, .L__ockl_hostcall_preview.numbered_sgpr)
	.set .L__ockl_fprintf_append_string_n.num_named_barrier, max(0, .L__ockl_hostcall_preview.num_named_barrier)
	.set .L__ockl_fprintf_append_string_n.private_seg_size, 672+max(.L__ockl_hostcall_preview.private_seg_size)
	.set .L__ockl_fprintf_append_string_n.uses_vcc, or(1, .L__ockl_hostcall_preview.uses_vcc)
	.set .L__ockl_fprintf_append_string_n.uses_flat_scratch, or(0, .L__ockl_hostcall_preview.uses_flat_scratch)
	.set .L__ockl_fprintf_append_string_n.has_dyn_sized_stack, or(0, .L__ockl_hostcall_preview.has_dyn_sized_stack)
	.set .L__ockl_fprintf_append_string_n.has_recursion, or(0, .L__ockl_hostcall_preview.has_recursion)
	.set .L__ockl_fprintf_append_string_n.has_indirect_call, or(0, .L__ockl_hostcall_preview.has_indirect_call)
	.section	.AMDGPU.csdata,"",@progbits
; Function info:
; codeLenInByte = 8916
; TotalNumSgprs: 40
; NumVgprs: 36
; NumAgprs: 32
; TotalNumVgprs: 68
; ScratchSize: 956
; MemoryBound: 0
	.text
	.p2align	2                               ; -- Begin function __ockl_fprintf_append_args
	.type	__ockl_fprintf_append_args,@function
__ockl_fprintf_append_args:             ; @__ockl_fprintf_append_args
; %bb.0:
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
	s_mov_b32 s19, s33
	s_mov_b32 s33, s32
	s_xor_saveexec_b64 s[0:1], -1
	scratch_store_dword off, v24, s33 offset:4 ; 4-byte Folded Spill
	scratch_store_dword off, v25, s33 offset:8 ; 4-byte Folded Spill
	s_mov_b64 exec, s[0:1]
	s_add_i32 s32, s32, 16
	v_writelane_b32 v24, s30, 0
	s_nop 1
	v_writelane_b32 v24, s31, 1
	scratch_store_dword off, v2, s33        ; 4-byte Folded Spill
	v_mov_b32_e32 v18, v0
	scratch_load_dword v0, off, s33         ; 4-byte Folded Reload
	v_mov_b32_e32 v20, v15
                                        ; kill: def $vgpr21 killed $vgpr16 killed $exec
	v_mov_b32_e32 v20, v13
                                        ; kill: def $vgpr21 killed $vgpr14 killed $exec
	v_mov_b32_e32 v20, v11
                                        ; kill: def $vgpr21 killed $vgpr12 killed $exec
	v_mov_b32_e32 v20, v9
                                        ; kill: def $vgpr21 killed $vgpr10 killed $exec
	v_mov_b32_e32 v20, v7
                                        ; kill: def $vgpr21 killed $vgpr8 killed $exec
	v_mov_b32_e32 v20, v5
                                        ; kill: def $vgpr21 killed $vgpr6 killed $exec
	v_mov_b32_e32 v20, v3
                                        ; kill: def $vgpr21 killed $vgpr4 killed $exec
                                        ; kill: def $vgpr18 killed $vgpr18 def $vgpr18_vgpr19 killed $exec
	v_mov_b32_e32 v19, v1
	s_mov_b32 s0, 0
	v_cmp_eq_u32_e64 s[0:1], v17, s0
	v_mov_b32_e32 v2, v19
	s_mov_b64 s[2:3], 2
	s_mov_b32 s16, s3
	v_or_b32_e64 v1, v2, s16
	v_mov_b32_e32 v17, v18
                                        ; kill: def $sgpr2 killed $sgpr2 killed $sgpr2_sgpr3
	v_or_b32_e64 v18, v17, s2
                                        ; kill: def $vgpr18 killed $vgpr18 def $vgpr18_vgpr19 killed $exec
	v_mov_b32_e32 v19, v1
	v_mov_b32_e32 v1, v19
	v_cndmask_b32_e64 v1, v1, v2, s[0:1]
	v_mov_b32_e32 v2, v18
	v_cndmask_b32_e64 v18, v2, v17, s[0:1]
                                        ; kill: def $vgpr18 killed $vgpr18 def $vgpr18_vgpr19 killed $exec
	v_mov_b32_e32 v19, v1
	v_mov_b32_e32 v1, v19
	s_mov_b32 s0, 0xffffff1f
	s_mov_b32 s1, -1
	s_mov_b32 s2, s1
	v_and_b32_e64 v1, v1, s2
	v_mov_b32_e32 v2, v18
                                        ; kill: def $sgpr0 killed $sgpr0 killed $sgpr0_sgpr1
	v_and_b32_e64 v20, v2, s0
                                        ; kill: def $vgpr20 killed $vgpr20 def $vgpr20_vgpr21 killed $exec
	v_mov_b32_e32 v21, v1
	s_mov_b32 s0, 0
	v_mov_b32_e32 v2, 0
                                        ; kill: def $vgpr0 killed $vgpr0 def $vgpr0_vgpr1 killed $exec
	v_mov_b32_e32 v1, v2
	s_mov_b32 s0, 5
	s_waitcnt vmcnt(0)
	v_lshlrev_b64 v[18:19], s0, v[0:1]
	v_mov_b32_e32 v0, v21
	v_mov_b32_e32 v1, v19
	v_or_b32_e64 v0, v0, v1
	v_mov_b32_e32 v1, v20
	v_mov_b32_e32 v2, v18
	v_or_b32_e64 v18, v1, v2
                                        ; kill: def $vgpr18 killed $vgpr18 def $vgpr18_vgpr19 killed $exec
	v_mov_b32_e32 v19, v0
	v_mov_b32_e32 v1, v18
	s_mov_b32 s0, 32
                                        ; implicit-def: $vgpr25 : SGPR spill to VGPR lane
	v_writelane_b32 v25, s0, 0
	v_lshrrev_b64 v[18:19], s0, v[18:19]
	v_mov_b32_e32 v2, v18
	s_getpc_b64 s[0:1]
	s_add_u32 s0, s0, __ockl_hostcall_preview@rel32@lo+4
	s_addc_u32 s1, s1, __ockl_hostcall_preview@rel32@hi+12
	v_mov_b32_e32 v0, 2
	s_swappc_b64 s[30:31], s[0:1]
	v_readlane_b32 s0, v25, 0
                                        ; implicit-def: $sgpr1
                                        ; implicit-def: $sgpr2
	v_mov_b32_e32 v2, s1
                                        ; kill: def $vgpr2 killed $vgpr2 def $vgpr2_vgpr3 killed $exec
	v_mov_b32_e32 v3, v1
	v_lshrrev_b64 v[2:3], s0, v[2:3]
	v_mov_b32_e32 v1, v2
	v_readlane_b32 s30, v24, 0
	v_readlane_b32 s31, v24, 1
	s_mov_b32 s32, s33
	s_xor_saveexec_b64 s[0:1], -1
	scratch_load_dword v24, off, s33 offset:4 ; 4-byte Folded Reload
	scratch_load_dword v25, off, s33 offset:8 ; 4-byte Folded Reload
	s_mov_b64 exec, s[0:1]
	s_mov_b32 s33, s19
	s_waitcnt vmcnt(0)
	s_setpc_b64 s[30:31]
.Lfunc_end7:
	.size	__ockl_fprintf_append_args, .Lfunc_end7-__ockl_fprintf_append_args
                                        ; -- End function
	.set .L__ockl_fprintf_append_args.num_vgpr, max(26, .L__ockl_hostcall_preview.num_vgpr)
	.set .L__ockl_fprintf_append_args.num_agpr, max(0, .L__ockl_hostcall_preview.num_agpr)
	.set .L__ockl_fprintf_append_args.numbered_sgpr, max(34, .L__ockl_hostcall_preview.numbered_sgpr)
	.set .L__ockl_fprintf_append_args.num_named_barrier, max(0, .L__ockl_hostcall_preview.num_named_barrier)
	.set .L__ockl_fprintf_append_args.private_seg_size, 16+max(.L__ockl_hostcall_preview.private_seg_size)
	.set .L__ockl_fprintf_append_args.uses_vcc, or(1, .L__ockl_hostcall_preview.uses_vcc)
	.set .L__ockl_fprintf_append_args.uses_flat_scratch, or(0, .L__ockl_hostcall_preview.uses_flat_scratch)
	.set .L__ockl_fprintf_append_args.has_dyn_sized_stack, or(0, .L__ockl_hostcall_preview.has_dyn_sized_stack)
	.set .L__ockl_fprintf_append_args.has_recursion, or(0, .L__ockl_hostcall_preview.has_recursion)
	.set .L__ockl_fprintf_append_args.has_indirect_call, or(0, .L__ockl_hostcall_preview.has_indirect_call)
	.section	.AMDGPU.csdata,"",@progbits
; Function info:
; codeLenInByte = 436
; TotalNumSgprs: 40
; NumVgprs: 26
; NumAgprs: 32
; TotalNumVgprs: 60
; ScratchSize: 300
; MemoryBound: 0
	.text
	.hidden	__assert_fail                   ; -- Begin function __assert_fail
	.weak	__assert_fail
	.p2align	2
	.type	__assert_fail,@function
__assert_fail:                          ; @__assert_fail
; %bb.0:
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
	s_mov_b32 s42, s33
	s_mov_b32 s33, s32
	s_xor_saveexec_b64 s[0:1], -1
	scratch_store_dword off, v36, s33 offset:160 ; 4-byte Folded Spill
	scratch_store_dword off, v37, s33 offset:164 ; 4-byte Folded Spill
	scratch_store_dword off, v38, s33 offset:168 ; 4-byte Folded Spill
	s_mov_b64 exec, s[0:1]
	s_add_i32 s32, s32, 0xb0
	v_writelane_b32 v36, s30, 0
	s_nop 1
	v_writelane_b32 v36, s31, 1
	scratch_store_dword off, v31, s33 offset:152 ; 4-byte Folded Spill
	scratch_store_dword off, v6, s33 offset:148 ; 4-byte Folded Spill
	v_mov_b32_e32 v6, v5
	scratch_load_dword v5, off, s33 offset:148 ; 4-byte Folded Reload
	s_nop 0
	scratch_store_dword off, v6, s33 offset:144 ; 4-byte Folded Spill
	scratch_store_dword off, v3, s33 offset:140 ; 4-byte Folded Spill
	v_mov_b32_e32 v6, v2
	scratch_load_dword v2, off, s33 offset:144 ; 4-byte Folded Reload
	v_mov_b32_e32 v8, v0
	scratch_load_dword v0, off, s33 offset:140 ; 4-byte Folded Reload
                                        ; implicit-def: $vgpr38 : SGPR spill to VGPR lane
	v_writelane_b32 v38, s15, 0
	v_writelane_b32 v38, s14, 1
	v_writelane_b32 v38, s13, 2
	v_writelane_b32 v38, s12, 3
	v_writelane_b32 v38, s10, 4
	s_nop 1
	v_writelane_b32 v38, s11, 5
	v_writelane_b32 v38, s8, 6
	s_nop 1
	v_writelane_b32 v38, s9, 7
	v_writelane_b32 v38, s6, 8
	s_nop 1
	v_writelane_b32 v38, s7, 9
	v_writelane_b32 v38, s4, 10
	s_nop 1
	v_writelane_b32 v38, s5, 11
                                        ; kill: def $vgpr2 killed $vgpr2 def $vgpr2_vgpr3 killed $exec
	s_waitcnt vmcnt(4)
	v_mov_b32_e32 v3, v5
                                        ; kill: def $vgpr6 killed $vgpr6 def $vgpr6_vgpr7 killed $exec
	s_waitcnt vmcnt(0)
	v_mov_b32_e32 v7, v0
                                        ; kill: def $vgpr8 killed $vgpr8 def $vgpr8_vgpr9 killed $exec
	v_mov_b32_e32 v9, v1
	s_mov_b64 s[2:3], 0
	s_mov_b32 s25, s3
	v_writelane_b32 v38, s25, 12
	s_mov_b32 s26, -1
	v_writelane_b32 v38, s26, 13
	s_mov_b32 s1, s33
	s_cmp_lg_u32 s1, s26
	s_mov_b64 s[16:17], src_private_base
	s_mov_b32 s24, s17
	v_writelane_b32 v38, s24, 14
	s_cselect_b32 s0, s24, s25
	s_mov_b32 s23, s2
	v_writelane_b32 v38, s23, 15
	s_cselect_b32 s20, s1, s23
                                        ; kill: def $sgpr20 killed $sgpr20 def $sgpr20_sgpr21
	s_mov_b32 s21, s0
	s_mov_b64 s[0:1], s[20:21]
	v_writelane_b32 v38, s0, 16
	s_nop 1
	v_writelane_b32 v38, s1, 17
	s_add_i32 s0, s33, 8
	s_mov_b32 s1, s0
	s_cmp_lg_u32 s1, s26
	s_cselect_b32 s0, s24, s25
	s_cselect_b32 s18, s1, s23
                                        ; kill: def $sgpr18 killed $sgpr18 def $sgpr18_sgpr19
	s_mov_b32 s19, s0
	s_mov_b64 s[0:1], s[18:19]
	v_writelane_b32 v38, s0, 18
	s_nop 1
	v_writelane_b32 v38, s1, 19
	s_add_i32 s0, s33, 16
	s_mov_b32 s1, s0
	s_cmp_lg_u32 s1, s26
	s_cselect_b32 s0, s24, s25
	s_cselect_b32 s2, s1, s23
                                        ; kill: def $sgpr2 killed $sgpr2 def $sgpr2_sgpr3
	s_mov_b32 s3, s0
	s_mov_b64 s[0:1], s[2:3]
	v_writelane_b32 v38, s0, 20
	s_nop 1
	v_writelane_b32 v38, s1, 21
	s_add_i32 s1, s33, 24
	s_mov_b32 s0, s1
	s_cmp_lg_u32 s0, s26
	s_cselect_b32 s16, s24, s25
	s_cselect_b32 s0, s0, s23
                                        ; kill: def $sgpr0 killed $sgpr0 def $sgpr0_sgpr1
	s_mov_b32 s1, s16
	s_mov_b64 s[16:17], s[0:1]
	v_writelane_b32 v38, s16, 22
	s_nop 1
	v_writelane_b32 v38, s17, 23
	s_add_i32 s17, s33, 32
	s_mov_b32 s16, s17
	s_cmp_lg_u32 s16, s26
	s_cselect_b32 s22, s24, s25
	s_cselect_b32 s16, s16, s23
                                        ; kill: def $sgpr16 killed $sgpr16 def $sgpr16_sgpr17
	s_mov_b32 s17, s22
	s_mov_b64 s[28:29], s[16:17]
	v_writelane_b32 v38, s28, 24
	s_nop 1
	v_writelane_b32 v38, s29, 25
	s_add_i32 s22, s33, 0x50
	s_mov_b32 s27, s22
	s_cmp_lg_u32 s27, s26
	s_cselect_b32 s22, s24, s25
	s_cselect_b32 s28, s27, s23
                                        ; kill: def $sgpr28 killed $sgpr28 def $sgpr28_sgpr29
	s_mov_b32 s29, s22
	v_writelane_b32 v38, s28, 26
	s_nop 1
	v_writelane_b32 v38, s29, 27
	v_writelane_b32 v38, s28, 28
	s_nop 1
	v_writelane_b32 v38, s29, 29
	s_add_i32 s22, s33, 0x58
	s_mov_b32 s27, s22
	s_cmp_lg_u32 s27, s26
	s_cselect_b32 s22, s24, s25
	s_cselect_b32 s28, s27, s23
                                        ; kill: def $sgpr28 killed $sgpr28 def $sgpr28_sgpr29
	s_mov_b32 s29, s22
	v_writelane_b32 v38, s28, 30
	s_nop 1
	v_writelane_b32 v38, s29, 31
	v_writelane_b32 v38, s28, 32
	s_nop 1
	v_writelane_b32 v38, s29, 33
	s_add_i32 s22, s33, 0x60
	s_mov_b32 s27, s22
	s_cmp_lg_u32 s27, s26
	s_cselect_b32 s22, s24, s25
	s_cselect_b32 s28, s27, s23
                                        ; kill: def $sgpr28 killed $sgpr28 def $sgpr28_sgpr29
	s_mov_b32 s29, s22
	v_writelane_b32 v38, s28, 34
	s_nop 1
	v_writelane_b32 v38, s29, 35
	s_add_i32 s22, s33, 0x68
	s_mov_b32 s27, s22
	s_cmp_lg_u32 s27, s26
	s_cselect_b32 s22, s24, s25
	s_cselect_b32 s28, s27, s23
                                        ; kill: def $sgpr28 killed $sgpr28 def $sgpr28_sgpr29
	s_mov_b32 s29, s22
	v_writelane_b32 v38, s28, 36
	s_nop 1
	v_writelane_b32 v38, s29, 37
	s_add_i32 s22, s33, 0x70
	s_mov_b32 s27, s22
	s_cmp_lg_u32 s27, s26
	s_cselect_b32 s22, s24, s25
	s_cselect_b32 s28, s27, s23
                                        ; kill: def $sgpr28 killed $sgpr28 def $sgpr28_sgpr29
	s_mov_b32 s29, s22
	v_writelane_b32 v38, s28, 38
	s_nop 1
	v_writelane_b32 v38, s29, 39
	s_add_i32 s27, s33, 0x78
	s_mov_b32 s22, s27
	s_cmp_lg_u32 s22, s26
	s_cselect_b32 s24, s24, s25
	s_cselect_b32 s22, s22, s23
                                        ; kill: def $sgpr22 killed $sgpr22 def $sgpr22_sgpr23
	s_mov_b32 s23, s24
	v_writelane_b32 v38, s22, 40
	s_nop 1
	v_writelane_b32 v38, s23, 41
	s_or_saveexec_b64 s[40:41], -1
	scratch_store_dword off, v38, s33 offset:128 ; 4-byte Folded Spill
	s_mov_b64 exec, s[40:41]
	v_mov_b64_e32 v[0:1], s[20:21]
	flat_store_dwordx2 v[0:1], v[8:9]
	v_mov_b64_e32 v[0:1], s[18:19]
	flat_store_dwordx2 v[0:1], v[6:7]
	v_mov_b64_e32 v[0:1], s[2:3]
	flat_store_dword v[0:1], v4
	v_mov_b64_e32 v[0:1], s[0:1]
	flat_store_dwordx2 v[0:1], v[2:3]
	v_mov_b32_e32 v0, 0
	scratch_store_dword off, v0, s33 offset:136 ; 4-byte Folded Spill
	s_getpc_b64 s[0:1]
	s_add_u32 s0, s0, __const.__assert_fail.fmt@rel32@lo+35
	s_addc_u32 s1, s1, __const.__assert_fail.fmt@rel32@hi+43
	global_load_dwordx4 v[2:5], v0, s[0:1]
	s_getpc_b64 s[0:1]
	s_add_u32 s0, s0, __const.__assert_fail.fmt@rel32@lo+4
	s_addc_u32 s1, s1, __const.__assert_fail.fmt@rel32@hi+12
	s_load_dwordx4 s[0:3], s[0:1], 0x0
	s_getpc_b64 s[18:19]
	s_add_u32 s18, s18, __const.__assert_fail.fmt@rel32@lo+20
	s_addc_u32 s19, s19, __const.__assert_fail.fmt@rel32@hi+28
	s_load_dwordx4 s[20:23], s[18:19], 0x0
	v_mov_b64_e32 v[0:1], s[16:17]
	s_waitcnt vmcnt(0)
	flat_store_dwordx4 v[0:1], v[2:5] offset:31
	v_mov_b64_e32 v[0:1], s[16:17]
	s_waitcnt lgkmcnt(0)
	v_mov_b64_e32 v[2:3], s[20:21]
	v_mov_b64_e32 v[4:5], s[22:23]
	flat_store_dwordx4 v[0:1], v[2:5] offset:16
	v_mov_b64_e32 v[0:1], s[16:17]
	s_nop 0
	v_mov_b64_e32 v[4:5], s[2:3]
	v_mov_b64_e32 v[2:3], s[0:1]
	flat_store_dwordx4 v[0:1], v[2:5]
	s_getpc_b64 s[0:1]
	s_add_u32 s0, s0, __ockl_fprintf_stderr_begin@rel32@lo+4
	s_addc_u32 s1, s1, __ockl_fprintf_stderr_begin@rel32@hi+12
	s_swappc_b64 s[30:31], s[0:1]
	scratch_load_dword v2, off, s33 offset:136 ; 4-byte Folded Reload
	v_readlane_b32 s2, v38, 26
	v_readlane_b32 s3, v38, 27
	v_readlane_b32 s0, v38, 30
	v_readlane_b32 s1, v38, 31
	v_mov_b32_e32 v4, v0
                                        ; kill: def $vgpr4 killed $vgpr4 def $vgpr4_vgpr5 killed $exec
	v_mov_b32_e32 v5, v1
	v_mov_b64_e32 v[0:1], s[2:3]
	flat_store_dwordx2 v[0:1], v[4:5]
	v_mov_b64_e32 v[0:1], s[0:1]
	s_waitcnt vmcnt(0)
	flat_store_dword v[0:1], v2
; %bb.1:
	s_or_saveexec_b64 s[40:41], -1
	scratch_load_dword v38, off, s33 offset:128 ; 4-byte Folded Reload
	s_mov_b64 exec, s[40:41]
	s_waitcnt vmcnt(0)
	v_readlane_b32 s0, v38, 24
	v_readlane_b32 s1, v38, 25
	v_readlane_b32 s2, v38, 34
	v_readlane_b32 s3, v38, 35
	s_nop 1
	v_mov_b64_e32 v[0:1], s[2:3]
	v_mov_b64_e32 v[2:3], s[0:1]
	flat_store_dwordx2 v[0:1], v[2:3]
	s_mov_b64 s[0:1], 0
                                        ; implicit-def: $sgpr2_sgpr3
	v_writelane_b32 v38, s0, 42
	s_nop 1
	v_writelane_b32 v38, s1, 43
	s_or_saveexec_b64 s[40:41], -1
	scratch_store_dword off, v38, s33 offset:128 ; 4-byte Folded Spill
	s_mov_b64 exec, s[40:41]
.LBB8_2:                                ; =>This Inner Loop Header: Depth=1
	s_or_saveexec_b64 s[40:41], -1
	scratch_load_dword v38, off, s33 offset:128 ; 4-byte Folded Reload
	s_mov_b64 exec, s[40:41]
	s_waitcnt vmcnt(0)
	v_readlane_b32 s2, v38, 34
	v_readlane_b32 s3, v38, 35
	v_readlane_b32 s0, v38, 44
	v_readlane_b32 s1, v38, 45
	v_readlane_b32 s4, v38, 42
	v_readlane_b32 s5, v38, 43
	s_nop 0
	v_writelane_b32 v38, s4, 46
	s_nop 1
	v_writelane_b32 v38, s5, 47
	v_mov_b64_e32 v[0:1], s[2:3]
	flat_load_dwordx2 v[0:1], v[0:1]
	s_mov_b64 s[4:5], 1
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshl_add_u64 v[4:5], v[0:1], 0, s[4:5]
	v_mov_b64_e32 v[2:3], s[2:3]
	flat_store_dwordx2 v[2:3], v[4:5]
	flat_load_ubyte v0, v[0:1]
	s_mov_b32 s2, 0
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_cmp_ne_u16_e64 s[2:3], v0, s2
	s_mov_b64 s[4:5], -1
	s_or_b64 s[0:1], s[0:1], exec
	v_writelane_b32 v38, s0, 48
	s_nop 1
	v_writelane_b32 v38, s1, 49
	v_writelane_b32 v38, s0, 50
	s_nop 1
	v_writelane_b32 v38, s1, 51
	s_mov_b64 s[0:1], exec
	v_writelane_b32 v38, s0, 52
	s_nop 1
	v_writelane_b32 v38, s1, 53
	s_or_saveexec_b64 s[40:41], -1
	scratch_store_dword off, v38, s33 offset:128 ; 4-byte Folded Spill
	s_mov_b64 exec, s[40:41]
	s_and_b64 s[0:1], s[0:1], s[2:3]
	s_mov_b64 exec, s[0:1]
	s_cbranch_execz .LBB8_4
; %bb.3:                                ;   in Loop: Header=BB8_2 Depth=1
	s_or_saveexec_b64 s[40:41], -1
	scratch_load_dword v38, off, s33 offset:128 ; 4-byte Folded Reload
	s_mov_b64 exec, s[40:41]
	s_waitcnt vmcnt(0)
	v_readlane_b32 s0, v38, 48
	v_readlane_b32 s1, v38, 49
	s_mov_b64 s[2:3], 0
	s_andn2_b64 s[0:1], s[0:1], exec
	v_writelane_b32 v38, s0, 50
	s_nop 1
	v_writelane_b32 v38, s1, 51
	s_or_saveexec_b64 s[40:41], -1
	scratch_store_dword off, v38, s33 offset:128 ; 4-byte Folded Spill
	s_mov_b64 exec, s[40:41]
.LBB8_4:                                ;   in Loop: Header=BB8_2 Depth=1
	s_or_saveexec_b64 s[40:41], -1
	scratch_load_dword v38, off, s33 offset:128 ; 4-byte Folded Reload
	s_mov_b64 exec, s[40:41]
	s_waitcnt vmcnt(0)
	v_readlane_b32 s0, v38, 52
	v_readlane_b32 s1, v38, 53
	s_or_b64 exec, exec, s[0:1]
	v_readlane_b32 s4, v38, 46
	v_readlane_b32 s5, v38, 47
	v_readlane_b32 s2, v38, 50
	v_readlane_b32 s3, v38, 51
	s_mov_b64 s[0:1], s[2:3]
	s_and_b64 s[0:1], exec, s[0:1]
	s_or_b64 s[0:1], s[0:1], s[4:5]
	v_writelane_b32 v38, s2, 44
	s_nop 1
	v_writelane_b32 v38, s3, 45
	s_mov_b64 s[2:3], s[0:1]
	v_writelane_b32 v38, s2, 42
	s_nop 1
	v_writelane_b32 v38, s3, 43
	s_mov_b64 s[2:3], s[0:1]
	v_writelane_b32 v38, s2, 54
	s_nop 1
	v_writelane_b32 v38, s3, 55
	s_or_saveexec_b64 s[40:41], -1
	scratch_store_dword off, v38, s33 offset:128 ; 4-byte Folded Spill
	s_mov_b64 exec, s[40:41]
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB8_2
; %bb.5:
	s_or_saveexec_b64 s[40:41], -1
	scratch_load_dword v38, off, s33 offset:128 ; 4-byte Folded Reload
	s_mov_b64 exec, s[40:41]
	s_waitcnt vmcnt(0)
	v_readlane_b32 s0, v38, 54
	v_readlane_b32 s1, v38, 55
	s_or_b64 exec, exec, s[0:1]
; %bb.6:
	s_or_saveexec_b64 s[40:41], -1
	scratch_load_dword v38, off, s33 offset:128 ; 4-byte Folded Reload
	s_mov_b64 exec, s[40:41]
	s_waitcnt vmcnt(0)
	v_readlane_b32 s0, v38, 32
	v_readlane_b32 s1, v38, 33
	v_readlane_b32 s2, v38, 24
	v_readlane_b32 s3, v38, 25
	v_readlane_b32 s4, v38, 34
	v_readlane_b32 s5, v38, 35
	s_nop 1
	v_mov_b64_e32 v[0:1], s[4:5]
	flat_load_dword v0, v[0:1]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_sub_u32_e64 v2, v0, s2
	v_mov_b64_e32 v[0:1], s[0:1]
	flat_store_dword v[0:1], v2
; %bb.7:
	s_or_saveexec_b64 s[40:41], -1
	scratch_load_dword v38, off, s33 offset:128 ; 4-byte Folded Reload
	s_mov_b64 exec, s[40:41]
	s_waitcnt vmcnt(0)
	v_readlane_b32 s16, v38, 28
	v_readlane_b32 s17, v38, 29
	v_readlane_b32 s15, v38, 0
	v_readlane_b32 s14, v38, 1
	v_readlane_b32 s13, v38, 2
	v_readlane_b32 s12, v38, 3
	v_readlane_b32 s10, v38, 4
	v_readlane_b32 s11, v38, 5
	v_readlane_b32 s8, v38, 6
	v_readlane_b32 s9, v38, 7
	v_readlane_b32 s6, v38, 8
	v_readlane_b32 s7, v38, 9
	v_readlane_b32 s4, v38, 10
	v_readlane_b32 s5, v38, 11
	v_readlane_b32 s0, v38, 24
	v_readlane_b32 s1, v38, 25
	v_readlane_b32 s2, v38, 32
	v_readlane_b32 s3, v38, 33
	scratch_load_dword v31, off, s33 offset:152 ; 4-byte Folded Reload
	v_mov_b64_e32 v[0:1], s[16:17]
	flat_load_dwordx2 v[2:3], v[0:1]
	v_mov_b64_e32 v[0:1], s[2:3]
	flat_load_dword v4, v[0:1]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_ashrrev_i32_e64 v0, 31, v4
	v_mov_b32_e32 v6, v4
	v_mov_b32_e32 v7, v0
	s_mov_b32 s3, 32
	s_lshr_b64 s[16:17], s[0:1], s3
	s_mov_b32 s2, s16
	v_lshrrev_b64 v[0:1], s3, v[2:3]
	v_mov_b32_e32 v1, v0
	v_lshrrev_b64 v[6:7], s3, v[6:7]
	v_mov_b32_e32 v5, v6
	s_mov_b32 s3, s0
	v_mov_b32_e32 v0, v2
	s_getpc_b64 s[0:1]
	s_add_u32 s0, s0, __ockl_fprintf_append_string_n@rel32@lo+4
	s_addc_u32 s1, s1, __ockl_fprintf_append_string_n@rel32@hi+12
	v_mov_b32_e32 v6, 0
	v_mov_b32_e32 v2, s3
	v_mov_b32_e32 v3, s2
	s_swappc_b64 s[30:31], s[0:1]
	v_readlane_b32 s0, v38, 28
	v_readlane_b32 s1, v38, 29
	v_mov_b32_e32 v2, v0
                                        ; kill: def $vgpr2 killed $vgpr2 def $vgpr2_vgpr3 killed $exec
	v_mov_b32_e32 v3, v1
	v_mov_b64_e32 v[0:1], s[0:1]
	flat_store_dwordx2 v[0:1], v[2:3]
; %bb.8:
	s_or_saveexec_b64 s[40:41], -1
	scratch_load_dword v38, off, s33 offset:128 ; 4-byte Folded Reload
	s_mov_b64 exec, s[40:41]
	s_waitcnt vmcnt(0)
	v_readlane_b32 s0, v38, 36
	v_readlane_b32 s1, v38, 37
	v_readlane_b32 s2, v38, 18
	v_readlane_b32 s3, v38, 19
	s_nop 1
	v_mov_b64_e32 v[0:1], s[2:3]
	flat_load_dwordx2 v[2:3], v[0:1]
	v_mov_b64_e32 v[0:1], s[0:1]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	flat_store_dwordx2 v[0:1], v[2:3]
	s_mov_b64 s[0:1], 0
                                        ; implicit-def: $sgpr2_sgpr3
	v_writelane_b32 v38, s0, 56
	s_nop 1
	v_writelane_b32 v38, s1, 57
	s_or_saveexec_b64 s[40:41], -1
	scratch_store_dword off, v38, s33 offset:128 ; 4-byte Folded Spill
	s_mov_b64 exec, s[40:41]
.LBB8_9:                                ; =>This Inner Loop Header: Depth=1
	s_or_saveexec_b64 s[40:41], -1
	scratch_load_dword v38, off, s33 offset:128 ; 4-byte Folded Reload
	s_mov_b64 exec, s[40:41]
	s_waitcnt vmcnt(0)
	v_readlane_b32 s2, v38, 36
	v_readlane_b32 s3, v38, 37
	v_readlane_b32 s0, v38, 58
	v_readlane_b32 s1, v38, 59
	v_readlane_b32 s4, v38, 56
	v_readlane_b32 s5, v38, 57
	s_nop 0
	v_writelane_b32 v38, s4, 60
	s_nop 1
	v_writelane_b32 v38, s5, 61
	v_mov_b64_e32 v[0:1], s[2:3]
	flat_load_dwordx2 v[0:1], v[0:1]
	s_mov_b64 s[4:5], 1
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshl_add_u64 v[4:5], v[0:1], 0, s[4:5]
	v_mov_b64_e32 v[2:3], s[2:3]
	flat_store_dwordx2 v[2:3], v[4:5]
	flat_load_ubyte v0, v[0:1]
	s_mov_b32 s2, 0
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_cmp_ne_u16_e64 s[2:3], v0, s2
	s_mov_b64 s[4:5], -1
	s_or_b64 s[0:1], s[0:1], exec
	v_writelane_b32 v38, s0, 62
	s_nop 1
	v_writelane_b32 v38, s1, 63
	s_or_saveexec_b64 s[40:41], -1
	scratch_store_dword off, v38, s33 offset:128 ; 4-byte Folded Spill
	s_mov_b64 exec, s[40:41]
                                        ; implicit-def: $vgpr38 : SGPR spill to VGPR lane
	v_writelane_b32 v38, s0, 0
	s_nop 1
	v_writelane_b32 v38, s1, 1
	s_mov_b64 s[0:1], exec
	v_writelane_b32 v38, s0, 2
	s_nop 1
	v_writelane_b32 v38, s1, 3
	s_or_saveexec_b64 s[40:41], -1
	scratch_store_dword off, v38, s33 offset:132 ; 4-byte Folded Spill
	s_mov_b64 exec, s[40:41]
	s_and_b64 s[0:1], s[0:1], s[2:3]
	s_mov_b64 exec, s[0:1]
	s_cbranch_execz .LBB8_11
; %bb.10:                               ;   in Loop: Header=BB8_9 Depth=1
	s_or_saveexec_b64 s[40:41], -1
	scratch_load_dword v37, off, s33 offset:128 ; 4-byte Folded Reload
	s_mov_b64 exec, s[40:41]
	s_waitcnt vmcnt(0)
	v_readlane_b32 s0, v37, 62
	v_readlane_b32 s1, v37, 63
	s_or_saveexec_b64 s[40:41], -1
	scratch_load_dword v38, off, s33 offset:132 ; 4-byte Folded Reload
	s_mov_b64 exec, s[40:41]
	s_mov_b64 s[2:3], 0
	s_andn2_b64 s[0:1], s[0:1], exec
	s_waitcnt vmcnt(0)
	v_writelane_b32 v38, s0, 0
	s_nop 1
	v_writelane_b32 v38, s1, 1
	s_or_saveexec_b64 s[40:41], -1
	scratch_store_dword off, v38, s33 offset:132 ; 4-byte Folded Spill
	s_mov_b64 exec, s[40:41]
.LBB8_11:                               ;   in Loop: Header=BB8_9 Depth=1
	s_or_saveexec_b64 s[40:41], -1
	scratch_load_dword v37, off, s33 offset:128 ; 4-byte Folded Reload
	s_mov_b64 exec, s[40:41]
	s_or_saveexec_b64 s[40:41], -1
	scratch_load_dword v38, off, s33 offset:132 ; 4-byte Folded Reload
	s_mov_b64 exec, s[40:41]
	s_waitcnt vmcnt(0)
	v_readlane_b32 s0, v38, 2
	v_readlane_b32 s1, v38, 3
	s_or_b64 exec, exec, s[0:1]
	v_readlane_b32 s4, v37, 60
	v_readlane_b32 s5, v37, 61
	v_readlane_b32 s2, v38, 0
	v_readlane_b32 s3, v38, 1
	s_mov_b64 s[0:1], s[2:3]
	s_and_b64 s[0:1], exec, s[0:1]
	s_or_b64 s[0:1], s[0:1], s[4:5]
	v_writelane_b32 v37, s2, 58
	s_nop 1
	v_writelane_b32 v37, s3, 59
	s_mov_b64 s[2:3], s[0:1]
	v_writelane_b32 v37, s2, 56
	s_nop 1
	v_writelane_b32 v37, s3, 57
	s_or_saveexec_b64 s[40:41], -1
	scratch_store_dword off, v37, s33 offset:128 ; 4-byte Folded Spill
	s_mov_b64 exec, s[40:41]
	s_mov_b64 s[2:3], s[0:1]
	v_writelane_b32 v38, s2, 4
	s_nop 1
	v_writelane_b32 v38, s3, 5
	s_or_saveexec_b64 s[40:41], -1
	scratch_store_dword off, v38, s33 offset:132 ; 4-byte Folded Spill
	s_mov_b64 exec, s[40:41]
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB8_9
; %bb.12:
	s_or_saveexec_b64 s[40:41], -1
	scratch_load_dword v38, off, s33 offset:132 ; 4-byte Folded Reload
	s_mov_b64 exec, s[40:41]
	s_waitcnt vmcnt(0)
	v_readlane_b32 s0, v38, 4
	v_readlane_b32 s1, v38, 5
	s_or_b64 exec, exec, s[0:1]
; %bb.13:
	s_or_saveexec_b64 s[40:41], -1
	scratch_load_dword v38, off, s33 offset:128 ; 4-byte Folded Reload
	s_mov_b64 exec, s[40:41]
	s_waitcnt vmcnt(0)
	v_readlane_b32 s0, v38, 32
	v_readlane_b32 s1, v38, 33
	v_readlane_b32 s2, v38, 18
	v_readlane_b32 s3, v38, 19
	v_readlane_b32 s4, v38, 36
	v_readlane_b32 s5, v38, 37
	s_nop 1
	v_mov_b64_e32 v[0:1], s[4:5]
	flat_load_dword v0, v[0:1]
	v_mov_b64_e32 v[2:3], s[2:3]
	flat_load_dword v1, v[2:3]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_sub_u32_e64 v2, v0, v1
	v_mov_b64_e32 v[0:1], s[0:1]
	flat_store_dword v[0:1], v2
; %bb.14:
	s_or_saveexec_b64 s[40:41], -1
	scratch_load_dword v38, off, s33 offset:128 ; 4-byte Folded Reload
	s_mov_b64 exec, s[40:41]
	s_waitcnt vmcnt(0)
	v_readlane_b32 s16, v38, 28
	v_readlane_b32 s17, v38, 29
	v_readlane_b32 s15, v38, 0
	v_readlane_b32 s14, v38, 1
	v_readlane_b32 s13, v38, 2
	v_readlane_b32 s12, v38, 3
	v_readlane_b32 s10, v38, 4
	v_readlane_b32 s11, v38, 5
	v_readlane_b32 s8, v38, 6
	v_readlane_b32 s9, v38, 7
	v_readlane_b32 s6, v38, 8
	v_readlane_b32 s7, v38, 9
	v_readlane_b32 s4, v38, 10
	v_readlane_b32 s5, v38, 11
	v_readlane_b32 s0, v38, 32
	v_readlane_b32 s1, v38, 33
	v_readlane_b32 s2, v38, 18
	v_readlane_b32 s3, v38, 19
	s_or_saveexec_b64 s[40:41], -1
	scratch_load_dword v37, off, s33 offset:132 ; 4-byte Folded Reload
	s_mov_b64 exec, s[40:41]
	scratch_load_dword v31, off, s33 offset:152 ; 4-byte Folded Reload
	v_mov_b64_e32 v[0:1], s[16:17]
	flat_load_dwordx2 v[8:9], v[0:1]
	v_mov_b64_e32 v[0:1], s[2:3]
	flat_load_dwordx2 v[6:7], v[0:1]
	v_mov_b64_e32 v[0:1], s[0:1]
	flat_load_dword v4, v[0:1]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_ashrrev_i32_e64 v0, 31, v4
	v_mov_b32_e32 v10, v4
	v_mov_b32_e32 v11, v0
	s_mov_b32 s0, 32
	v_writelane_b32 v37, s0, 6
	s_or_saveexec_b64 s[40:41], -1
	scratch_store_dword off, v37, s33 offset:132 ; 4-byte Folded Spill
	s_mov_b64 exec, s[40:41]
	v_lshrrev_b64 v[0:1], s0, v[8:9]
	v_mov_b32_e32 v1, v0
	v_lshrrev_b64 v[2:3], s0, v[6:7]
	v_mov_b32_e32 v3, v2
	v_lshrrev_b64 v[10:11], s0, v[10:11]
	v_mov_b32_e32 v5, v10
	v_mov_b32_e32 v0, v8
	v_mov_b32_e32 v2, v6
	s_getpc_b64 s[0:1]
	s_add_u32 s0, s0, __ockl_fprintf_append_string_n@rel32@lo+4
	s_addc_u32 s1, s1, __ockl_fprintf_append_string_n@rel32@hi+12
	v_mov_b32_e32 v6, 0
	scratch_store_dword off, v6, s33 offset:156 ; 4-byte Folded Spill
	s_swappc_b64 s[30:31], s[0:1]
	scratch_load_dword v31, off, s33 offset:152 ; 4-byte Folded Reload
	scratch_load_dword v17, off, s33 offset:156 ; 4-byte Folded Reload
	v_readlane_b32 s2, v38, 20
	v_readlane_b32 s3, v38, 21
	v_readlane_b32 s0, v37, 6
	v_readlane_b32 s4, v38, 10
	v_readlane_b32 s5, v38, 11
	v_readlane_b32 s6, v38, 8
	v_readlane_b32 s7, v38, 9
	v_readlane_b32 s8, v38, 6
	v_readlane_b32 s9, v38, 7
	v_readlane_b32 s10, v38, 4
	v_readlane_b32 s11, v38, 5
	v_readlane_b32 s12, v38, 3
	v_readlane_b32 s13, v38, 2
	v_readlane_b32 s14, v38, 1
	v_readlane_b32 s15, v38, 0
	v_readlane_b32 s16, v38, 28
	v_readlane_b32 s17, v38, 29
	v_mov_b32_e32 v2, v0
                                        ; kill: def $vgpr2 killed $vgpr2 def $vgpr2_vgpr3 killed $exec
	v_mov_b32_e32 v3, v1
	v_mov_b64_e32 v[0:1], s[16:17]
	flat_store_dwordx2 v[0:1], v[2:3]
	v_mov_b64_e32 v[0:1], s[16:17]
	flat_load_dwordx2 v[4:5], v[0:1]
	v_mov_b64_e32 v[0:1], s[2:3]
	flat_load_dword v3, v[0:1]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshrrev_b64 v[0:1], s0, v[4:5]
	v_mov_b32_e32 v1, v0
	v_mov_b32_e32 v0, v4
	s_getpc_b64 s[0:1]
	s_add_u32 s0, s0, __ockl_fprintf_append_args@rel32@lo+4
	s_addc_u32 s1, s1, __ockl_fprintf_append_args@rel32@hi+12
	v_mov_b32_e32 v2, 1
	v_mov_b32_e32 v4, v17
	v_mov_b32_e32 v5, v17
	v_mov_b32_e32 v6, v17
	v_mov_b32_e32 v7, v17
	v_mov_b32_e32 v8, v17
	v_mov_b32_e32 v9, v17
	v_mov_b32_e32 v10, v17
	v_mov_b32_e32 v11, v17
	v_mov_b32_e32 v12, v17
	v_mov_b32_e32 v13, v17
	v_mov_b32_e32 v14, v17
	v_mov_b32_e32 v15, v17
	v_mov_b32_e32 v16, v17
	s_swappc_b64 s[30:31], s[0:1]
	v_readlane_b32 s0, v38, 28
	v_readlane_b32 s1, v38, 29
	v_mov_b32_e32 v2, v0
                                        ; kill: def $vgpr2 killed $vgpr2 def $vgpr2_vgpr3 killed $exec
	v_mov_b32_e32 v3, v1
	v_mov_b64_e32 v[0:1], s[0:1]
	flat_store_dwordx2 v[0:1], v[2:3]
; %bb.15:
	s_or_saveexec_b64 s[40:41], -1
	scratch_load_dword v37, off, s33 offset:128 ; 4-byte Folded Reload
	s_mov_b64 exec, s[40:41]
	s_waitcnt vmcnt(0)
	v_readlane_b32 s0, v37, 38
	v_readlane_b32 s1, v37, 39
	v_readlane_b32 s2, v37, 22
	v_readlane_b32 s3, v37, 23
	s_or_saveexec_b64 s[40:41], -1
	scratch_load_dword v38, off, s33 offset:132 ; 4-byte Folded Reload
	s_mov_b64 exec, s[40:41]
	v_mov_b64_e32 v[0:1], s[2:3]
	flat_load_dwordx2 v[2:3], v[0:1]
	v_mov_b64_e32 v[0:1], s[0:1]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	flat_store_dwordx2 v[0:1], v[2:3]
	s_mov_b64 s[0:1], 0
                                        ; implicit-def: $sgpr2_sgpr3
	v_writelane_b32 v38, s0, 7
	s_nop 1
	v_writelane_b32 v38, s1, 8
	s_or_saveexec_b64 s[40:41], -1
	scratch_store_dword off, v38, s33 offset:132 ; 4-byte Folded Spill
	s_mov_b64 exec, s[40:41]
.LBB8_16:                               ; =>This Inner Loop Header: Depth=1
	s_or_saveexec_b64 s[40:41], -1
	scratch_load_dword v37, off, s33 offset:128 ; 4-byte Folded Reload
	s_mov_b64 exec, s[40:41]
	s_or_saveexec_b64 s[40:41], -1
	scratch_load_dword v38, off, s33 offset:132 ; 4-byte Folded Reload
	s_mov_b64 exec, s[40:41]
	s_waitcnt vmcnt(0)
	v_readlane_b32 s2, v37, 38
	v_readlane_b32 s3, v37, 39
	v_readlane_b32 s0, v38, 9
	v_readlane_b32 s1, v38, 10
	v_readlane_b32 s4, v38, 7
	v_readlane_b32 s5, v38, 8
	s_nop 0
	v_writelane_b32 v38, s4, 11
	s_nop 1
	v_writelane_b32 v38, s5, 12
	v_mov_b64_e32 v[0:1], s[2:3]
	flat_load_dwordx2 v[0:1], v[0:1]
	s_mov_b64 s[4:5], 1
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshl_add_u64 v[4:5], v[0:1], 0, s[4:5]
	v_mov_b64_e32 v[2:3], s[2:3]
	flat_store_dwordx2 v[2:3], v[4:5]
	flat_load_ubyte v0, v[0:1]
	s_mov_b32 s2, 0
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_cmp_ne_u16_e64 s[2:3], v0, s2
	s_mov_b64 s[4:5], -1
	s_or_b64 s[0:1], s[0:1], exec
	v_writelane_b32 v38, s0, 13
	s_nop 1
	v_writelane_b32 v38, s1, 14
	v_writelane_b32 v38, s0, 15
	s_nop 1
	v_writelane_b32 v38, s1, 16
	s_mov_b64 s[0:1], exec
	v_writelane_b32 v38, s0, 17
	s_nop 1
	v_writelane_b32 v38, s1, 18
	s_or_saveexec_b64 s[40:41], -1
	scratch_store_dword off, v38, s33 offset:132 ; 4-byte Folded Spill
	s_mov_b64 exec, s[40:41]
	s_and_b64 s[0:1], s[0:1], s[2:3]
	s_mov_b64 exec, s[0:1]
	s_cbranch_execz .LBB8_18
; %bb.17:                               ;   in Loop: Header=BB8_16 Depth=1
	s_or_saveexec_b64 s[40:41], -1
	scratch_load_dword v38, off, s33 offset:132 ; 4-byte Folded Reload
	s_mov_b64 exec, s[40:41]
	s_waitcnt vmcnt(0)
	v_readlane_b32 s0, v38, 13
	v_readlane_b32 s1, v38, 14
	s_mov_b64 s[2:3], 0
	s_andn2_b64 s[0:1], s[0:1], exec
	v_writelane_b32 v38, s0, 15
	s_nop 1
	v_writelane_b32 v38, s1, 16
	s_or_saveexec_b64 s[40:41], -1
	scratch_store_dword off, v38, s33 offset:132 ; 4-byte Folded Spill
	s_mov_b64 exec, s[40:41]
.LBB8_18:                               ;   in Loop: Header=BB8_16 Depth=1
	s_or_saveexec_b64 s[40:41], -1
	scratch_load_dword v38, off, s33 offset:132 ; 4-byte Folded Reload
	s_mov_b64 exec, s[40:41]
	s_waitcnt vmcnt(0)
	v_readlane_b32 s0, v38, 17
	v_readlane_b32 s1, v38, 18
	s_or_b64 exec, exec, s[0:1]
	v_readlane_b32 s4, v38, 11
	v_readlane_b32 s5, v38, 12
	v_readlane_b32 s2, v38, 15
	v_readlane_b32 s3, v38, 16
	s_mov_b64 s[0:1], s[2:3]
	s_and_b64 s[0:1], exec, s[0:1]
	s_or_b64 s[0:1], s[0:1], s[4:5]
	v_writelane_b32 v38, s2, 9
	s_nop 1
	v_writelane_b32 v38, s3, 10
	s_mov_b64 s[2:3], s[0:1]
	v_writelane_b32 v38, s2, 7
	s_nop 1
	v_writelane_b32 v38, s3, 8
	s_mov_b64 s[2:3], s[0:1]
	v_writelane_b32 v38, s2, 19
	s_nop 1
	v_writelane_b32 v38, s3, 20
	s_or_saveexec_b64 s[40:41], -1
	scratch_store_dword off, v38, s33 offset:132 ; 4-byte Folded Spill
	s_mov_b64 exec, s[40:41]
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB8_16
; %bb.19:
	s_or_saveexec_b64 s[40:41], -1
	scratch_load_dword v38, off, s33 offset:132 ; 4-byte Folded Reload
	s_mov_b64 exec, s[40:41]
	s_waitcnt vmcnt(0)
	v_readlane_b32 s0, v38, 19
	v_readlane_b32 s1, v38, 20
	s_or_b64 exec, exec, s[0:1]
; %bb.20:
	s_or_saveexec_b64 s[40:41], -1
	scratch_load_dword v38, off, s33 offset:128 ; 4-byte Folded Reload
	s_mov_b64 exec, s[40:41]
	s_waitcnt vmcnt(0)
	v_readlane_b32 s0, v38, 32
	v_readlane_b32 s1, v38, 33
	v_readlane_b32 s2, v38, 22
	v_readlane_b32 s3, v38, 23
	v_readlane_b32 s4, v38, 38
	v_readlane_b32 s5, v38, 39
	s_nop 1
	v_mov_b64_e32 v[0:1], s[4:5]
	flat_load_dword v0, v[0:1]
	v_mov_b64_e32 v[2:3], s[2:3]
	flat_load_dword v1, v[2:3]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_sub_u32_e64 v2, v0, v1
	v_mov_b64_e32 v[0:1], s[0:1]
	flat_store_dword v[0:1], v2
; %bb.21:
	s_or_saveexec_b64 s[40:41], -1
	scratch_load_dword v38, off, s33 offset:128 ; 4-byte Folded Reload
	s_mov_b64 exec, s[40:41]
	s_waitcnt vmcnt(0)
	v_readlane_b32 s16, v38, 28
	v_readlane_b32 s17, v38, 29
	v_readlane_b32 s15, v38, 0
	v_readlane_b32 s14, v38, 1
	v_readlane_b32 s13, v38, 2
	v_readlane_b32 s12, v38, 3
	v_readlane_b32 s10, v38, 4
	v_readlane_b32 s11, v38, 5
	v_readlane_b32 s8, v38, 6
	v_readlane_b32 s9, v38, 7
	v_readlane_b32 s6, v38, 8
	v_readlane_b32 s7, v38, 9
	v_readlane_b32 s4, v38, 10
	v_readlane_b32 s5, v38, 11
	v_readlane_b32 s0, v38, 32
	v_readlane_b32 s1, v38, 33
	v_readlane_b32 s2, v38, 22
	v_readlane_b32 s3, v38, 23
	scratch_load_dword v31, off, s33 offset:152 ; 4-byte Folded Reload
	v_mov_b64_e32 v[0:1], s[16:17]
	flat_load_dwordx2 v[8:9], v[0:1]
	v_mov_b64_e32 v[0:1], s[2:3]
	flat_load_dwordx2 v[6:7], v[0:1]
	v_mov_b64_e32 v[0:1], s[0:1]
	flat_load_dword v4, v[0:1]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_ashrrev_i32_e64 v0, 31, v4
	v_mov_b32_e32 v10, v4
	v_mov_b32_e32 v11, v0
	s_mov_b32 s0, 32
	v_lshrrev_b64 v[0:1], s0, v[8:9]
	v_mov_b32_e32 v1, v0
	v_lshrrev_b64 v[2:3], s0, v[6:7]
	v_mov_b32_e32 v3, v2
	v_lshrrev_b64 v[10:11], s0, v[10:11]
	v_mov_b32_e32 v5, v10
	v_mov_b32_e32 v0, v8
	v_mov_b32_e32 v2, v6
	s_getpc_b64 s[0:1]
	s_add_u32 s0, s0, __ockl_fprintf_append_string_n@rel32@lo+4
	s_addc_u32 s1, s1, __ockl_fprintf_append_string_n@rel32@hi+12
	v_mov_b32_e32 v6, 0
	s_swappc_b64 s[30:31], s[0:1]
	v_readlane_b32 s0, v38, 28
	v_readlane_b32 s1, v38, 29
	v_mov_b32_e32 v2, v0
                                        ; kill: def $vgpr2 killed $vgpr2 def $vgpr2_vgpr3 killed $exec
	v_mov_b32_e32 v3, v1
	v_mov_b64_e32 v[0:1], s[0:1]
	flat_store_dwordx2 v[0:1], v[2:3]
; %bb.22:
	s_or_saveexec_b64 s[40:41], -1
	scratch_load_dword v37, off, s33 offset:128 ; 4-byte Folded Reload
	s_mov_b64 exec, s[40:41]
	s_waitcnt vmcnt(0)
	v_readlane_b32 s0, v37, 40
	v_readlane_b32 s1, v37, 41
	v_readlane_b32 s2, v37, 16
	v_readlane_b32 s3, v37, 17
	s_or_saveexec_b64 s[40:41], -1
	scratch_load_dword v38, off, s33 offset:132 ; 4-byte Folded Reload
	s_mov_b64 exec, s[40:41]
	v_mov_b64_e32 v[0:1], s[2:3]
	flat_load_dwordx2 v[2:3], v[0:1]
	v_mov_b64_e32 v[0:1], s[0:1]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	flat_store_dwordx2 v[0:1], v[2:3]
	s_mov_b64 s[0:1], 0
                                        ; implicit-def: $sgpr2_sgpr3
	v_writelane_b32 v38, s0, 21
	s_nop 1
	v_writelane_b32 v38, s1, 22
	s_or_saveexec_b64 s[40:41], -1
	scratch_store_dword off, v38, s33 offset:132 ; 4-byte Folded Spill
	s_mov_b64 exec, s[40:41]
.LBB8_23:                               ; =>This Inner Loop Header: Depth=1
	s_or_saveexec_b64 s[40:41], -1
	scratch_load_dword v37, off, s33 offset:128 ; 4-byte Folded Reload
	s_mov_b64 exec, s[40:41]
	s_or_saveexec_b64 s[40:41], -1
	scratch_load_dword v38, off, s33 offset:132 ; 4-byte Folded Reload
	s_mov_b64 exec, s[40:41]
	s_waitcnt vmcnt(0)
	v_readlane_b32 s2, v37, 40
	v_readlane_b32 s3, v37, 41
	v_readlane_b32 s0, v38, 23
	v_readlane_b32 s1, v38, 24
	v_readlane_b32 s4, v38, 21
	v_readlane_b32 s5, v38, 22
	s_nop 0
	v_writelane_b32 v38, s4, 25
	s_nop 1
	v_writelane_b32 v38, s5, 26
	v_mov_b64_e32 v[0:1], s[2:3]
	flat_load_dwordx2 v[0:1], v[0:1]
	s_mov_b64 s[4:5], 1
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshl_add_u64 v[4:5], v[0:1], 0, s[4:5]
	v_mov_b64_e32 v[2:3], s[2:3]
	flat_store_dwordx2 v[2:3], v[4:5]
	flat_load_ubyte v0, v[0:1]
	s_mov_b32 s2, 0
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_cmp_ne_u16_e64 s[2:3], v0, s2
	s_mov_b64 s[4:5], -1
	s_or_b64 s[0:1], s[0:1], exec
	v_writelane_b32 v38, s0, 27
	s_nop 1
	v_writelane_b32 v38, s1, 28
	v_writelane_b32 v38, s0, 29
	s_nop 1
	v_writelane_b32 v38, s1, 30
	s_mov_b64 s[0:1], exec
	v_writelane_b32 v38, s0, 31
	s_nop 1
	v_writelane_b32 v38, s1, 32
	s_or_saveexec_b64 s[40:41], -1
	scratch_store_dword off, v38, s33 offset:132 ; 4-byte Folded Spill
	s_mov_b64 exec, s[40:41]
	s_and_b64 s[0:1], s[0:1], s[2:3]
	s_mov_b64 exec, s[0:1]
	s_cbranch_execz .LBB8_25
; %bb.24:                               ;   in Loop: Header=BB8_23 Depth=1
	s_or_saveexec_b64 s[40:41], -1
	scratch_load_dword v38, off, s33 offset:132 ; 4-byte Folded Reload
	s_mov_b64 exec, s[40:41]
	s_waitcnt vmcnt(0)
	v_readlane_b32 s0, v38, 27
	v_readlane_b32 s1, v38, 28
	s_mov_b64 s[2:3], 0
	s_andn2_b64 s[0:1], s[0:1], exec
	v_writelane_b32 v38, s0, 29
	s_nop 1
	v_writelane_b32 v38, s1, 30
	s_or_saveexec_b64 s[40:41], -1
	scratch_store_dword off, v38, s33 offset:132 ; 4-byte Folded Spill
	s_mov_b64 exec, s[40:41]
.LBB8_25:                               ;   in Loop: Header=BB8_23 Depth=1
	s_or_saveexec_b64 s[40:41], -1
	scratch_load_dword v38, off, s33 offset:132 ; 4-byte Folded Reload
	s_mov_b64 exec, s[40:41]
	s_waitcnt vmcnt(0)
	v_readlane_b32 s0, v38, 31
	v_readlane_b32 s1, v38, 32
	s_or_b64 exec, exec, s[0:1]
	v_readlane_b32 s4, v38, 25
	v_readlane_b32 s5, v38, 26
	v_readlane_b32 s2, v38, 29
	v_readlane_b32 s3, v38, 30
	s_mov_b64 s[0:1], s[2:3]
	s_and_b64 s[0:1], exec, s[0:1]
	s_or_b64 s[0:1], s[0:1], s[4:5]
	v_writelane_b32 v38, s2, 23
	s_nop 1
	v_writelane_b32 v38, s3, 24
	s_mov_b64 s[2:3], s[0:1]
	v_writelane_b32 v38, s2, 21
	s_nop 1
	v_writelane_b32 v38, s3, 22
	s_mov_b64 s[2:3], s[0:1]
	v_writelane_b32 v38, s2, 33
	s_nop 1
	v_writelane_b32 v38, s3, 34
	s_or_saveexec_b64 s[40:41], -1
	scratch_store_dword off, v38, s33 offset:132 ; 4-byte Folded Spill
	s_mov_b64 exec, s[40:41]
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB8_23
; %bb.26:
	s_or_saveexec_b64 s[40:41], -1
	scratch_load_dword v38, off, s33 offset:132 ; 4-byte Folded Reload
	s_mov_b64 exec, s[40:41]
	s_waitcnt vmcnt(0)
	v_readlane_b32 s0, v38, 33
	v_readlane_b32 s1, v38, 34
	s_or_b64 exec, exec, s[0:1]
; %bb.27:
	s_or_saveexec_b64 s[40:41], -1
	scratch_load_dword v38, off, s33 offset:128 ; 4-byte Folded Reload
	s_mov_b64 exec, s[40:41]
	s_waitcnt vmcnt(0)
	v_readlane_b32 s0, v38, 32
	v_readlane_b32 s1, v38, 33
	v_readlane_b32 s2, v38, 16
	v_readlane_b32 s3, v38, 17
	v_readlane_b32 s4, v38, 40
	v_readlane_b32 s5, v38, 41
	s_nop 1
	v_mov_b64_e32 v[0:1], s[4:5]
	flat_load_dword v0, v[0:1]
	v_mov_b64_e32 v[2:3], s[2:3]
	flat_load_dword v1, v[2:3]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_sub_u32_e64 v2, v0, v1
	v_mov_b64_e32 v[0:1], s[0:1]
	flat_store_dword v[0:1], v2
; %bb.28:
	s_or_saveexec_b64 s[40:41], -1
	scratch_load_dword v38, off, s33 offset:128 ; 4-byte Folded Reload
	s_mov_b64 exec, s[40:41]
	s_waitcnt vmcnt(0)
	v_readlane_b32 s15, v38, 0
	v_readlane_b32 s14, v38, 1
	v_readlane_b32 s13, v38, 2
	v_readlane_b32 s12, v38, 3
	v_readlane_b32 s10, v38, 4
	v_readlane_b32 s11, v38, 5
	v_readlane_b32 s8, v38, 6
	v_readlane_b32 s9, v38, 7
	v_readlane_b32 s6, v38, 8
	v_readlane_b32 s7, v38, 9
	v_readlane_b32 s4, v38, 10
	v_readlane_b32 s5, v38, 11
	v_readlane_b32 s0, v38, 32
	v_readlane_b32 s1, v38, 33
	v_readlane_b32 s2, v38, 16
	v_readlane_b32 s3, v38, 17
	v_readlane_b32 s16, v38, 28
	v_readlane_b32 s17, v38, 29
	scratch_load_dword v31, off, s33 offset:152 ; 4-byte Folded Reload
	s_nop 0
	v_mov_b64_e32 v[0:1], s[16:17]
	flat_load_dwordx2 v[8:9], v[0:1]
	v_mov_b64_e32 v[0:1], s[2:3]
	flat_load_dwordx2 v[6:7], v[0:1]
	v_mov_b64_e32 v[0:1], s[0:1]
	flat_load_dword v4, v[0:1]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_ashrrev_i32_e64 v0, 31, v4
	v_mov_b32_e32 v10, v4
	v_mov_b32_e32 v11, v0
	s_mov_b32 s0, 32
	v_lshrrev_b64 v[0:1], s0, v[8:9]
	v_mov_b32_e32 v1, v0
	v_lshrrev_b64 v[2:3], s0, v[6:7]
	v_mov_b32_e32 v3, v2
	v_lshrrev_b64 v[10:11], s0, v[10:11]
	v_mov_b32_e32 v5, v10
	v_mov_b32_e32 v0, v8
	v_mov_b32_e32 v2, v6
	s_getpc_b64 s[0:1]
	s_add_u32 s0, s0, __ockl_fprintf_append_string_n@rel32@lo+4
	s_addc_u32 s1, s1, __ockl_fprintf_append_string_n@rel32@hi+12
	v_mov_b32_e32 v6, 1
	s_swappc_b64 s[30:31], s[0:1]
	s_trap 2
	v_readlane_b32 s30, v36, 0
	v_readlane_b32 s31, v36, 1
	s_mov_b32 s32, s33
	s_xor_saveexec_b64 s[0:1], -1
	scratch_load_dword v36, off, s33 offset:160 ; 4-byte Folded Reload
	scratch_load_dword v37, off, s33 offset:164 ; 4-byte Folded Reload
	scratch_load_dword v38, off, s33 offset:168 ; 4-byte Folded Reload
	s_mov_b64 exec, s[0:1]
	s_mov_b32 s33, s42
	s_waitcnt vmcnt(0)
	s_setpc_b64 s[30:31]
.Lfunc_end8:
	.size	__assert_fail, .Lfunc_end8-__assert_fail
                                        ; -- End function
	.set __assert_fail.num_vgpr, max(39, .L__ockl_fprintf_stderr_begin.num_vgpr, .L__ockl_fprintf_append_string_n.num_vgpr, .L__ockl_fprintf_append_args.num_vgpr)
	.set __assert_fail.num_agpr, max(0, .L__ockl_fprintf_stderr_begin.num_agpr, .L__ockl_fprintf_append_string_n.num_agpr, .L__ockl_fprintf_append_args.num_agpr)
	.set __assert_fail.numbered_sgpr, max(43, .L__ockl_fprintf_stderr_begin.numbered_sgpr, .L__ockl_fprintf_append_string_n.numbered_sgpr, .L__ockl_fprintf_append_args.numbered_sgpr)
	.set __assert_fail.num_named_barrier, max(0, .L__ockl_fprintf_stderr_begin.num_named_barrier, .L__ockl_fprintf_append_string_n.num_named_barrier, .L__ockl_fprintf_append_args.num_named_barrier)
	.set __assert_fail.private_seg_size, 176+max(.L__ockl_fprintf_stderr_begin.private_seg_size, .L__ockl_fprintf_append_string_n.private_seg_size, .L__ockl_fprintf_append_args.private_seg_size)
	.set __assert_fail.uses_vcc, or(1, .L__ockl_fprintf_stderr_begin.uses_vcc, .L__ockl_fprintf_append_string_n.uses_vcc, .L__ockl_fprintf_append_args.uses_vcc)
	.set __assert_fail.uses_flat_scratch, or(0, .L__ockl_fprintf_stderr_begin.uses_flat_scratch, .L__ockl_fprintf_append_string_n.uses_flat_scratch, .L__ockl_fprintf_append_args.uses_flat_scratch)
	.set __assert_fail.has_dyn_sized_stack, or(0, .L__ockl_fprintf_stderr_begin.has_dyn_sized_stack, .L__ockl_fprintf_append_string_n.has_dyn_sized_stack, .L__ockl_fprintf_append_args.has_dyn_sized_stack)
	.set __assert_fail.has_recursion, or(0, .L__ockl_fprintf_stderr_begin.has_recursion, .L__ockl_fprintf_append_string_n.has_recursion, .L__ockl_fprintf_append_args.has_recursion)
	.set __assert_fail.has_indirect_call, or(0, .L__ockl_fprintf_stderr_begin.has_indirect_call, .L__ockl_fprintf_append_string_n.has_indirect_call, .L__ockl_fprintf_append_args.has_indirect_call)
	.section	.AMDGPU.csdata,"",@progbits
; Function info:
; codeLenInByte = 6236
; TotalNumSgprs: 49
; NumVgprs: 39
; NumAgprs: 32
; TotalNumVgprs: 72
; ScratchSize: 1132
; MemoryBound: 0
	.text
	.hidden	__assertfail                    ; -- Begin function __assertfail
	.weak	__assertfail
	.p2align	2
	.type	__assertfail,@function
__assertfail:                           ; @__assertfail
; %bb.0:
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
	s_mov_b32 s0, s33
	s_mov_b32 s33, s32
	s_trap 2
	s_mov_b32 s33, s0
	s_setpc_b64 s[30:31]
.Lfunc_end9:
	.size	__assertfail, .Lfunc_end9-__assertfail
                                        ; -- End function
	.set __assertfail.num_vgpr, 0
	.set __assertfail.num_agpr, 0
	.set __assertfail.numbered_sgpr, 34
	.set __assertfail.num_named_barrier, 0
	.set __assertfail.private_seg_size, 0
	.set __assertfail.uses_vcc, 0
	.set __assertfail.uses_flat_scratch, 0
	.set __assertfail.has_dyn_sized_stack, 0
	.set __assertfail.has_recursion, 0
	.set __assertfail.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Function info:
; codeLenInByte = 24
; TotalNumSgprs: 40
; NumVgprs: 0
; NumAgprs: 0
; TotalNumVgprs: 0
; ScratchSize: 0
; MemoryBound: 0
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 39
	.set amdgpu.max_num_agpr, 32
	.set amdgpu.max_num_sgpr, 43
	.text
	.type	__const.__assert_fail.fmt,@object ; @__const.__assert_fail.fmt
	.section	.rodata.str1.16,"aMS",@progbits,1
	.p2align	4, 0x0
__const.__assert_fail.fmt:
	.asciz	"%s:%u: %s: Device-side assertion `%s' failed.\n"
	.size	__const.__assert_fail.fmt, 47

	.type	__hip_cuid_cb17ed47c5977a5f,@object ; @__hip_cuid_cb17ed47c5977a5f
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_cb17ed47c5977a5f
__hip_cuid_cb17ed47c5977a5f:
	.byte	0                               ; 0x0
	.size	__hip_cuid_cb17ed47c5977a5f, 1

	.type	__oclc_ISA_version,@object      ; @__oclc_ISA_version
	.section	.rodata,"a",@progbits
	.p2align	2, 0x0
__oclc_ISA_version:
	.long	9402                            ; 0x24ba
	.size	__oclc_ISA_version, 4

	.type	__oclc_ABI_version,@object      ; @__oclc_ABI_version
	.p2align	2, 0x0
__oclc_ABI_version:
	.long	600                             ; 0x258
	.size	__oclc_ABI_version, 4

	.ident	"AMD clang version 22.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.2.0 26014 7b800a19466229b8479a78de19143dc33c3ab9b5)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __ockl_fprintf_stderr_begin
	.addrsig_sym __ockl_fprintf_append_args
	.addrsig_sym __ockl_fprintf_append_string_n
	.addrsig_sym __hip_cuid_cb17ed47c5977a5f
	.amdgpu_metadata
---
amdhsa.kernels:  []
amdhsa.target:   amdgcn-amd-amdhsa--gfx942
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata

# __CLANG_OFFLOAD_BUNDLE____END__ hip-amdgcn-amd-amdhsa--gfx942

# __CLANG_OFFLOAD_BUNDLE____START__ host-x86_64-unknown-linux-gnu-
	.file	"main.hip"
                                        # Start of file scope inline assembly
	.globl	_ZSt21ios_base_library_initv

                                        # End of file scope inline assembly
	.text
	.globl	main                            # -- Begin function main
	.p2align	4
	.type	main,@function
main:                                   # @main
	.cfi_startproc
# %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$1472, %rsp                     # imm = 0x5C0
	leaq	-1472(%rbp), %rdi
	xorl	%esi, %esi
	callq	hipGetDevicePropertiesR0600
	movabsq	$_ZSt4cout, %rdi
	movabsq	$.L.str, %rsi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	movq	%rax, %rdi
	leaq	-1472(%rbp), %rsi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	movq	%rax, %rdi
	movabsq	$.L.str.1, %rsi
	callq	_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	xorl	%eax, %eax
	addq	$1472, %rsp                     # imm = 0x5C0
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end0:
	.size	main, .Lfunc_end0-main
	.cfi_endproc
                                        # -- End function
	.type	.L.str,@object                  # @.str
	.section	.rodata.str1.1,"aMS",@progbits,1
.L.str:
	.asciz	"Info: running on device"
	.size	.L.str, 24

	.type	.L.str.1,@object                # @.str.1
.L.str.1:
	.asciz	"\n"
	.size	.L.str.1, 2

	.type	__hip_cuid_cb17ed47c5977a5f,@object # @__hip_cuid_cb17ed47c5977a5f
	.bss
	.globl	__hip_cuid_cb17ed47c5977a5f
__hip_cuid_cb17ed47c5977a5f:
	.byte	0                               # 0x0
	.size	__hip_cuid_cb17ed47c5977a5f, 1

	.ident	"AMD clang version 22.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.2.0 26014 7b800a19466229b8479a78de19143dc33c3ab9b5)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym hipGetDevicePropertiesR0600
	.addrsig_sym _ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	.addrsig_sym _ZSt4cout
	.addrsig_sym __hip_cuid_cb17ed47c5977a5f

# __CLANG_OFFLOAD_BUNDLE____END__ host-x86_64-unknown-linux-gnu-
