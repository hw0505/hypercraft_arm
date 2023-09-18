// Copyright (c) 2023 Beihang University, Huawei Technologies Co.,Ltd. All rights reserved.
// Rust-Shyper is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
//          http://license.coscl.org.cn/MulanPSL2
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
// See the Mulan PSL v2 for more details.

use alloc::sync::Arc;
use alloc::vec::Vec;
use page_table::PagingIf;
use core::mem::size_of;
use spin::Mutex;

// type ContextFrame = crate::arch::contextFrame::Aarch64ContextFrame;
use cortex_a::registers::*;
use tock_registers::interfaces::*;
 
use crate::arch::{ContextFrame, memcpy_safe};
use crate::arch::contextFrame::VmContext;
use crate::traits::ContextFrameTrait;
use crate::arch::vm::Vm;
use crate::arch::gic::{GICD, GICC, GICH};
use crate::arch::cpu::CpuState;

#[derive(Clone, Copy, Debug)]
pub enum VcpuState {
    VcpuInv = 0,
    VcpuPend = 1,
    VcpuAct = 2,
}

#[derive(Clone)]
pub struct Vcpu {
    pub inner: Arc<Mutex<VcpuInner>>,
}

impl Vcpu {
    pub fn new(id:usize, phys_id: usize) -> Vcpu {
        Vcpu {
            inner: Arc::new(Mutex::new(VcpuInner::new(id, phys_id))),
        }
    }

    pub fn set_gicc_ctlr(&self, ctlr: u32) {
        let mut inner = self.inner.lock();
        inner.vm_ctx.gic_state.saved_ctlr = ctlr;
    }

    pub fn set_hcr(&self, hcr: u64) {
        let mut inner = self.inner.lock();
        inner.vm_ctx.hcr_el2 = hcr;
    }

    pub fn state(&self) -> VcpuState {
        let inner = self.inner.lock();
        inner.state.clone()
    }

    pub fn set_state(&self, state: VcpuState) {
        let mut inner = self.inner.lock();
        inner.state = state;
    }

    pub fn id(&self) -> usize {
        let inner = self.inner.lock();
        inner.id
    }

    pub fn vm(&self) -> Option<Vm> {
        let inner = self.inner.lock();
        inner.vm.clone()
    }

    pub fn phys_id(&self) -> usize {
        let inner = self.inner.lock();
        inner.phys_id
    }

    pub fn vm_id(&self) -> usize {
        self.vm().unwrap().id()
    }

    pub fn reset_context(&self) {
        let mut inner = self.inner.lock();
        inner.reset_context();
    }

    pub fn context_ext_regs_store(&self) {
        let mut inner = self.inner.lock();
        inner.context_ext_regs_store();
    }

    pub fn vcpu_ctx_addr(&self) -> usize {
        let inner = self.inner.lock();
        inner.vcpu_ctx_addr()
    }

    pub fn set_elr(&self, elr: usize) {
        let mut inner = self.inner.lock();
        inner.set_elr(elr);
    }

    pub fn elr(&self) -> usize {
        let inner = self.inner.lock();
        inner.vcpu_ctx.exception_pc()
    }

    pub fn set_gpr(&self, idx: usize, val: usize) {
        let mut inner = self.inner.lock();
        inner.set_gpr(idx, val);
    }

    pub fn show_ctx(&self) {
        let inner = self.inner.lock();
        inner.show_ctx();
    }

    pub fn push_int(&self, int: usize) {
        let mut inner = self.inner.lock();
        if !inner.int_list.contains(&int) {
            inner.int_list.push(int);
        }
    }
}

pub struct VcpuInner {
    pub id: usize,
    pub phys_id: usize,
    pub state: VcpuState,
    pub vm: Option<Vm>,
    pub int_list: Vec<usize>,
    pub vcpu_ctx: ContextFrame,
    pub vm_ctx: VmContext,
}

impl VcpuInner {
    pub fn new(id: usize, phys_id: usize) -> VcpuInner {
        VcpuInner {
            id: id,
            phys_id: phys_id,
            state: VcpuState::VcpuInv,
            vm: None,
            int_list: vec![],
            vcpu_ctx: ContextFrame::default(),
            vm_ctx: VmContext::default(),
        }
    }

    fn vcpu_ctx_addr(&self) -> usize {
        &(self.vcpu_ctx) as *const _ as usize
    }

    fn vm_id(&self) -> usize {
        let vm = self.vm.as_ref().unwrap();
        vm.id()
    }

    fn arch_ctx_reset(&mut self) {
        // let migrate = self.vm.as_ref().unwrap().migration_state();
        // if !migrate {
        self.vm_ctx.cntvoff_el2 = 0;
        self.vm_ctx.sctlr_el1 = 0x30C50830;
        self.vm_ctx.cntkctl_el1 = 0;
        self.vm_ctx.pmcr_el0 = 0;
        self.vm_ctx.vtcr_el2 = 0x8001355c;
        // }
        let mut vmpidr = 0;
        vmpidr |= 1 << 31;

        vmpidr |= self.id;
        self.vm_ctx.vmpidr_el2 = vmpidr as u64;
    }

    fn reset_vtimer_offset(&mut self) {
        let curpct = cortex_a::registers::CNTPCT_EL0.get() as u64;
        self.vm_ctx.cntvoff_el2 = curpct - self.vm_ctx.cntvct_el0;
    }
    
    fn reset_context(&mut self) {
        self.arch_ctx_reset();
        self.gic_ctx_reset();
    }

    fn gic_ctx_reset(&mut self) {
        if let Some(gich) = GICH {
            for i in 0..gich.get_lrs_num() {
            self.vm_ctx.gic_state.saved_lr[i] = 0;
            }
        } else {
            info!("No available gich in gic_ctx_reset")
        }
        self.vm_ctx.gic_state.saved_hcr |= 1 << 2;
    }

    fn context_ext_regs_store(&mut self) {
        self.vm_ctx.ext_regs_store();
    }

    fn reset_vm_ctx(&mut self) {
        self.vm_ctx.reset();
    }

    fn set_elr(&mut self, elr: usize) {
        self.vcpu_ctx.set_exception_pc(elr);
    }

    fn set_gpr(&mut self, idx: usize, val: usize) {
        self.vcpu_ctx.set_gpr(idx, val);
    }

    fn show_ctx(&self) {
        info!(
            "cntvoff_el2 {:x}, sctlr_el1 {:x}, cntkctl_el1 {:x}, pmcr_el0 {:x}, vtcr_el2 {:x} x0 {:x}",
            self.vm_ctx.cntvoff_el2,
            self.vm_ctx.sctlr_el1,
            self.vm_ctx.cntkctl_el1,
            self.vm_ctx.pmcr_el0,
            self.vm_ctx.vtcr_el2,
            self.vcpu_ctx.gpr(0)
        );
    }
}

pub static VCPU_LIST: Mutex<Vec<Vcpu>> = Mutex::new(Vec::new());
/* 
pub fn restore_vcpu_gic(cur_vcpu: Option<Vcpu>, trgt_vcpu: Vcpu) {
    // println!("restore_vcpu_gic");
    match cur_vcpu {
        None => {
            // println!("None cur vmid trgt {}", trgt_vcpu.vm_id());
            trgt_vcpu.gic_restore_context();
        }
        Some(active_vcpu) => {
            if trgt_vcpu.vm_id() != active_vcpu.vm_id() {
                // println!("different vm_id cur {}, trgt {}", active_vcpu.vm_id(), trgt_vcpu.vm_id());
                active_vcpu.gic_save_context();
                trgt_vcpu.gic_restore_context();
            }
        }
    }
}

pub fn save_vcpu_gic(cur_vcpu: Option<Vcpu>, trgt_vcpu: Vcpu) {
    // println!("save_vcpu_gic");
    match cur_vcpu {
        None => {
            trgt_vcpu.gic_save_context();
        }
        Some(active_vcpu) => {
            if trgt_vcpu.vm_id() != active_vcpu.vm_id() {
                trgt_vcpu.gic_save_context();
                active_vcpu.gic_restore_context();
            }
        }
    }
}


pub fn vcpu_arch_init(vm: Vm, vcpu: Vcpu) {
    let config = vm.config();
    let mut vcpu_inner = vcpu.inner.lock();
    vcpu_inner.vcpu_ctx.set_argument(config.device_tree_load_ipa());
    vcpu_inner.vcpu_ctx.set_exception_pc(config.kernel_entry_point());
    vcpu_inner.vcpu_ctx.spsr =
        (SPSR_EL1::M::EL1h + SPSR_EL1::I::Masked + SPSR_EL1::F::Masked + SPSR_EL1::A::Masked + SPSR_EL1::D::Masked)
            .value;
}
*/
/* 
pub fn vcpu_alloc() -> Option<Vcpu> {
    let mut vcpu_list = VCPU_LIST.lock();
    if vcpu_list.len() >= 8 {
        return None;
    }
    let val = Vcpu::default();
    vcpu_list.push(val.clone());
    Some(val)
}

pub fn vcpu_remove(vcpu: Vcpu) {
    let mut vcpu_list = VCPU_LIST.lock();
    for (idx, core) in vcpu_list.iter().enumerate() {
        if core.id() == vcpu.id() && core.vm_id() == vcpu.vm_id() {
            vcpu_list.remove(idx);
            return;
        }
    }
    panic!("illegal vm{} vcpu{}, not exist in vcpu_list", vcpu.vm_id(), vcpu.id());
}

// WARNING: No Auto `drop` in this function

pub fn vcpu_run(announce: bool) -> ! {
    {
        let vcpu = current_cpu().active_vcpu.clone().unwrap();
        let vm = vcpu.vm().unwrap();

        current_cpu().cpu_state = CpuState::CpuRun;
        vm_interface_set_state(active_vm_id(), VmState::VmActive);

        vcpu.context_vm_restore();
    }
    extern "C" {
        fn context_vm_entry(ctx: usize) -> !;
    }
    unsafe {
        context_vm_entry(current_cpu().context_addr.unwrap());
    }
}
*/



#[derive(Default)]
/// A virtual CPU within a guest
pub struct VCpu<H: HyperCraftHal> {
    //vcpu_id: usize,
    //regs: VmCpuRegisters,
    // gpt: G,
    // pub guest: Arc<Guest>,

    pub id: usize,
    pub phys_id: usize,
    pub state: VcpuState,
    pub vm: Option<Vm>,
    pub int_list: Vec<usize>,
    pub vcpu_ctx: ContextFrame,
    pub vm_ctx: VmContext,

    marker: PhantomData<H>,
}

impl<H: HyperCraftHal> VCpu<H> {

    // /// Create a new vCPU
    // pub fn new(vcpu_id: usize, entry: GuestPhysAddr) -> Self {
    //     Self {
    //         vcpu_id,
    //         phys_id: 0,
    //         state: VcpuState::VcpuInv,
    //         vm: None,
    //         int_list: vec![],
    //         vcpu_ctx: ContextFrame::default(),
    //         vm_ctx: VmContext::default(),
    //         marker: PhantomData,
    //     }
    // }


    /// Create a new vCPU
    pub fn new(vcpu_id: usize, entry: GuestPhysAddr) -> Self {
        // let mut regs = VmCpuRegisters::default();
        // // Set hstatus
        // let mut hstatus = LocalRegisterCopy::<usize, hstatus::Register>::new(
        //     riscv::register::hstatus::read().bits(),
        // );
        // hstatus.modify(hstatus::spv::Supervisor);
        // // Set SPVP bit in order to accessing VS-mode memory from HS-mode.
        // hstatus.modify(hstatus::spvp::Supervisor);
        // CSR.hstatus.write_value(hstatus.get());
        // regs.guest_regs.hstatus = hstatus.get();

        // // Set sstatus
        // let mut sstatus = sstatus::read();
        // sstatus.set_spp(sstatus::SPP::Supervisor);
        // regs.guest_regs.sstatus = sstatus.bits();

        // regs.guest_regs.gprs.set_reg(GprIndex::A0, 0);
        // regs.guest_regs.gprs.set_reg(GprIndex::A1, 0x9000_0000);

        // // Set entry
        // regs.guest_regs.sepc = entry;

        let mut vcpu_ctx = ContextFrame::default();
        let mut vm_ctx = VmContext::default();
        let arg = &config.memory_region()[0]; //need to fix


        vcpu_ctx.set_argument(arg.ipa_start + arg.length); //need to fix
        //vcpu_ctx.set_exception_pc(config.kernel_entry_point()); //need to fix
        vcpu_ctx.set_exception_pc(entry);
        vcpu_ctx.spsr =
        (SPSR_EL1::M::EL1h + SPSR_EL1::I::Masked + SPSR_EL1::F::Masked + SPSR_EL1::A::Masked + SPSR_EL1::D::Masked)
            .value;

        //arch_ctx_reset
        vm_ctx.cntvoff_el2 = 0;
        vm_ctx.sctlr_el1 = 0x30C50830;
        vm_ctx.cntkctl_el1 = 0;
        vm_ctx.pmcr_el0 = 0;
        vm_ctx.vtcr_el2 = 0x8001355c;
        // }
        let mut vmpidr = 0;
        vmpidr |= 1 << 31;

        // #[cfg(feature = "tx2")]
        // if self.vm_id() == 0 {
        //     // A57 is cluster #1 for L4T
        //     vmpidr |= 0x100;
        // }

        vmpidr |= vcpu_id;
        vm_ctx.vmpidr_el2 = vmpidr as u64;

        //gic_ctx_reset
        use crate::arch::gich_lrs_num;
        for i in 0..gich_lrs_num() {
            vm_ctx.gic_state.lr[i] = 0;
        }
        vm_ctx.gic_state.hcr |= 1 << 2;

        
        Self {
            vcpu_id,
            phys_id: 0,
            state: VcpuState::VcpuInv,
            vm: None,
            int_list: vec![],
            vcpu_ctx: vcpu_ctx,
            vm_ctx: vm_ctx,
            marker: PhantomData,
        }
    }

    fn arch_ctx_reset(&mut self) {
        // let migrate = self.vm.as_ref().unwrap().migration_state();
        // if !migrate {
        self.vm_ctx.cntvoff_el2 = 0;
        self.vm_ctx.sctlr_el1 = 0x30C50830;
        self.vm_ctx.cntkctl_el1 = 0;
        self.vm_ctx.pmcr_el0 = 0;
        self.vm_ctx.vtcr_el2 = 0x8001355c;
        // }
        let mut vmpidr = 0;
        vmpidr |= 1 << 31;

        #[cfg(feature = "tx2")]
        if self.vm_id() == 0 {
            // A57 is cluster #1 for L4T
            vmpidr |= 0x100;
        }

        vmpidr |= self.id;
        self.vm_ctx.vmpidr_el2 = vmpidr as u64;
    }

    fn reset_context(&mut self) {
        // let migrate = self.vm.as_ref().unwrap().migration_state();
        self.arch_ctx_reset();
        // if !migrate {
        self.gic_ctx_reset();
        // }
        use crate::kernel::vm_if_get_type;
        match vm_if_get_type(self.vm_id()) {
            VmType::VmTBma => {
                println!("vm {} bma ctx restore", self.vm_id());
                self.reset_vm_ctx();
                self.context_ext_regs_store();
            }
            _ => {}
        }
    }

    fn gic_ctx_reset(&mut self) {
        use crate::arch::gich_lrs_num;
        for i in 0..gich_lrs_num() {
            self.vm_ctx.gic_state.lr[i] = 0;
        }
        self.vm_ctx.gic_state.hcr |= 1 << 2;
    }

    fn context_ext_regs_store(&mut self) {
        self.vm_ctx.ext_regs_store();
    }

    fn reset_vm_ctx(&mut self) {
        self.vm_ctx.reset();
    }

    fn set_elr(&mut self, elr: usize) {
        self.vcpu_ctx.set_exception_pc(elr);
    }

    fn set_gpr(&mut self, idx: usize, val: usize) {
        self.vcpu_ctx.set_gpr(idx, val);
    }

    fn show_ctx(&self) {
        println!(
            "cntvoff_el2 {:x}, sctlr_el1 {:x}, cntkctl_el1 {:x}, pmcr_el0 {:x}, vtcr_el2 {:x} x0 {:x}",
            self.vm_ctx.cntvoff_el2,
            self.vm_ctx.sctlr_el1,
            self.vm_ctx.cntkctl_el1,
            self.vm_ctx.pmcr_el0,
            self.vm_ctx.vtcr_el2,
            self.vcpu_ctx.gpr(0)
        );
    }



    /// Initialize nested mmu.
    pub fn init_page_map(&mut self, token: usize) {
        // Set hgatp
        // TODO: Sv39 currently, but should be configurable
        self.regs.virtual_hs_csrs.hgatp = token;
        unsafe {
            core::arch::asm!(
                "csrw hgatp, {hgatp}",
                hgatp = in(reg) self.regs.virtual_hs_csrs.hgatp,
            );
            core::arch::riscv64::hfence_gvma_all();
        }
    }

    /// Restore vCPU registers from the guest's GPRs
    pub fn restore_gprs(&mut self, gprs: &GeneralPurposeRegisters) {
        for index in 0..32 {
            self.regs.guest_regs.gprs.set_reg(
                GprIndex::from_raw(index).unwrap(),
                gprs.reg(GprIndex::from_raw(index).unwrap()),
            )
        }
    }

    /// Save vCPU registers to the guest's GPRs
    pub fn save_gprs(&self, gprs: &mut GeneralPurposeRegisters) {
        for index in 0..32 {
            gprs.set_reg(
                GprIndex::from_raw(index).unwrap(),
                self.regs
                    .guest_regs
                    .gprs
                    .reg(GprIndex::from_raw(index).unwrap()),
            );
        }
    }

    /// Runs this vCPU until traps.
    pub fn run(&mut self) -> VmExitInfo {
        let regs = &mut self.regs;
        unsafe {
            // Safe to run the guest as it only touches memory assigned to it by being owned
            // by its page table
            _run_guest(regs);
        }
        // Save off the trap information
        regs.trap_csrs.scause = scause::read().bits();
        regs.trap_csrs.stval = stval::read();
        regs.trap_csrs.htval = htval::read();
        regs.trap_csrs.htinst = htinst::read();

        let scause = scause::read();
        use scause::{Exception, Interrupt, Trap};
        match scause.cause() {
            Trap::Exception(Exception::VirtualSupervisorEnvCall) => {
                let sbi_msg = SbiMessage::from_regs(regs.guest_regs.gprs.a_regs()).ok();
                VmExitInfo::Ecall(sbi_msg)
            }
            Trap::Interrupt(Interrupt::SupervisorTimer) => VmExitInfo::TimerInterruptEmulation,
            Trap::Interrupt(Interrupt::SupervisorExternal) => {
                VmExitInfo::ExternalInterruptEmulation
            }
            Trap::Exception(Exception::LoadGuestPageFault)
            | Trap::Exception(Exception::StoreGuestPageFault) => {
                let fault_addr = regs.trap_csrs.htval << 2 | regs.trap_csrs.stval & 0x3;
                // debug!(
                //     "fault_addr: {:#x}, htval: {:#x}, stval: {:#x}, sepc: {:#x}, scause: {:?}",
                //     fault_addr,
                //     regs.trap_csrs.htval,
                //     regs.trap_csrs.stval,
                //     regs.guest_regs.sepc,
                //     scause.cause()
                // );
                VmExitInfo::PageFault {
                    fault_addr,
                    // Note that this address is not necessarily guest virtual as the guest may or
                    // may not have 1st-stage translation enabled in VSATP. We still use GuestVirtAddr
                    // here though to distinguish it from addresses (e.g. in HTVAL, or passed via a
                    // TEECALL) which are exclusively guest-physical. Furthermore we only access guest
                    // instructions via the HLVX instruction, which will take the VSATP translation
                    // mode into account.
                    falut_pc: regs.guest_regs.sepc,
                    inst: regs.trap_csrs.htinst as u32,
                    priv_level: PrivilegeLevel::from_hstatus(regs.guest_regs.hstatus),
                }
            }
            _ => {
                panic!(
                    "Unhandled trap: {:?}, sepc: {:#x}, stval: {:#x}",
                    scause.cause(),
                    regs.guest_regs.sepc,
                    regs.trap_csrs.stval
                );
            }
        }
    }

    /// Gets one of the vCPU's general purpose registers.
    pub fn get_gpr(&self, index: GprIndex) -> usize {
        self.regs.guest_regs.gprs.reg(index)
    }

    /// Set one of the vCPU's general purpose register.
    pub fn set_gpr(&mut self, index: GprIndex, val: usize) {
        self.regs.guest_regs.gprs.set_reg(index, val);
    }

    /// Advance guest pc by `instr_len` bytes
    pub fn advance_pc(&mut self, instr_len: usize) {
        self.regs.guest_regs.sepc += instr_len
    }

    /// Gets the vCPU's id.
    pub fn vcpu_id(&self) -> usize {
        self.vcpu_id
    }

    /// Gets the vCPU's registers.
    pub fn regs(&mut self) -> &mut VmCpuRegisters {
        &mut self.regs
    }
}