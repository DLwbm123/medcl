"""Linux inference confinement: Landlock files and seccomp network/process limits.

Applied in a fresh, single-threaded child before any uploaded bytes are loaded.
Unsupported kernels or policy failures raise; callers never run without confinement.
"""
import ctypes
import ctypes.util
import errno
import os
from pathlib import Path
import platform


def restrict(read_paths, output, scratch):
    if platform.machine() not in ("x86_64", "aarch64"):
        raise RuntimeError("unsupported Landlock syscall architecture")
    library = ctypes.util.find_library("seccomp")
    if not library:
        raise RuntimeError("libseccomp unavailable")
    sec = ctypes.CDLL(library)
    libc = ctypes.CDLL(None, use_errno=True)
    class Ruleset(ctypes.Structure):
        _fields_ = [("handled_access_fs", ctypes.c_uint64)]
    class PathRule(ctypes.Structure):
        _pack_ = 1
        _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]
    abi = libc.syscall(444, 0, 0, 1)
    if abi < 1:
        raise RuntimeError("Landlock unavailable")
    # ABI 1 handles bits 0..12; deny refer/truncate as well when supported.
    handled = (1 << (15 if abi >= 3 else 14 if abi >= 2 else 13)) - 1
    ruleset = Ruleset(handled)
    fd = libc.syscall(444, ctypes.byref(ruleset), ctypes.sizeof(ruleset), 0)
    if fd < 0:
        raise OSError(ctypes.get_errno(), "Landlock ruleset")
    try:
        write = (1 << 1) | ((1 << 14) if abi >= 3 else 0)
        paths = [(Path(p).resolve(), (1 << 2) | (1 << 3)) for p in read_paths]
        paths += [(Path("/dev/null"), (1 << 1) | (1 << 2))]
        paths += [(Path(output).resolve(), write), (Path(scratch).resolve(), write | sum(1 << bit for bit in (2, 3, 4, 5, 7, 8)))]
        for path, access in paths:
            if not path.is_dir():
                access &= ~(1 << 3)
            parent = os.open(path, os.O_PATH | os.O_CLOEXEC)
            try:
                rule = PathRule(access, parent)
                if libc.syscall(445, fd, 1, ctypes.byref(rule), 0) != 0:
                    raise OSError(ctypes.get_errno(), "Landlock path rule")
            finally:
                os.close(parent)
        if libc.prctl(38, 1, 0, 0, 0) != 0 or libc.syscall(446, fd, 0) != 0:
            raise OSError(ctypes.get_errno(), "Landlock restriction")
    finally:
        os.close(fd)

    sec.seccomp_init.argtypes = [ctypes.c_uint32]
    sec.seccomp_init.restype = ctypes.c_void_p
    sec.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    sec.seccomp_syscall_resolve_name.restype = ctypes.c_int
    class Arg(ctypes.Structure):
        _fields_ = [("arg", ctypes.c_uint), ("op", ctypes.c_uint),
                    ("datum_a", ctypes.c_uint64), ("datum_b", ctypes.c_uint64)]
    sec.seccomp_rule_add_array.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int,
                                         ctypes.c_uint, ctypes.POINTER(Arg)]
    sec.seccomp_load.argtypes = [ctypes.c_void_p]
    sec.seccomp_release.argtypes = [ctypes.c_void_p]
    context = sec.seccomp_init(0x7fff0000)  # SCMP_ACT_ALLOW
    if not context:
        raise RuntimeError("seccomp initialization failed")
    def deny(name, error=errno.EPERM, comparison=None):
        number = sec.seccomp_syscall_resolve_name(name.encode("ascii"))
        if number < 0:
            return  # syscall does not exist on this architecture
        if sec.seccomp_rule_add_array(context, 0x50000 | error, number,
                                     1 if comparison else 0, ctypes.byref(comparison) if comparison else None) != 0:
            raise RuntimeError("seccomp rule failed: " + name)
    try:
        for name in ("socket", "socketpair", "connect", "bind", "listen", "accept", "accept4",
                     "execve", "execveat", "fork", "vfork", "ptrace", "process_vm_readv", "process_vm_writev",
                     "pidfd_getfd", "pidfd_send_signal", "kill", "tkill", "mount", "umount2", "pivot_root",
                     "chroot", "unshare", "setns", "open_by_handle_at", "bpf", "userfaultfd", "perf_event_open",
                     "io_uring_setup", "io_uring_enter", "io_uring_register", "truncate", "ftruncate",
                     "chmod", "fchmod", "fchmodat", "fchmodat2", "utime", "utimes", "futimesat", "utimensat",
                     "reboot", "swapon", "swapoff", "init_module", "finit_module", "delete_module",
                     "kexec_load", "kexec_file_load", "syslog", "acct", "quotactl", "settimeofday", "clock_settime", "chown", "fchown", "lchown", "fchownat",
                     "setxattr", "lsetxattr", "fsetxattr", "removexattr", "lremovexattr", "fremovexattr"):
            deny(name)
        deny("clone3", errno.ENOSYS)  # libc falls back to clone for threads
        deny("clone", comparison=Arg(0, 7, 0x10000, 0))  # MASKED_EQ: require CLONE_THREAD
        # Permit signals within the current thread group only.
        deny("tgkill", comparison=Arg(0, 1, os.getpid(), 0))  # NE
        if sec.seccomp_load(context) != 0:
            raise RuntimeError("seccomp activation failed")
    finally:
        sec.seccomp_release(context)
