//! SOCK_SEQPACKET client that sends a `Frame` plus an optional fd.
//!
//! SOCK_SEQPACKET over AF_UNIX preserves message boundaries (unlike
//! SOCK_STREAM): one sendmsg = exactly one recvmsg on the overlay side,
//! so we never have to prefix a length. The fd rides in ancillary data
//! via SCM_RIGHTS; the kernel dup()s it across the process boundary
//! and closes the original send-side reference when sendmsg returns.
//!
//! Framing note: for KIND_RELEASE we send the same struct but with no
//! ancillary fd. The overlay uses `msg.msg_controllen` to distinguish.

use std::io;
use std::mem;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
use std::path::Path;

use crate::error::{Error, Result};
use crate::protocol::Frame;

/// Thin wrapper around a ``SOCK_SEQPACKET`` AF_UNIX client socket.
///
/// std's ``UnixDatagram`` uses ``SOCK_DGRAM`` which is incompatible
/// with the SEQPACKET server the overlay binds. We open the socket
/// with ``socket(2)`` directly instead. SEQPACKET preserves message
/// boundaries (one sendmsg = one recvmsg) and guarantees ordering,
/// which is what our protocol wants: no framing in the payload, and
/// no partial-message handling.
pub struct FrameSocket {
    fd: OwnedFd,
}

impl FrameSocket {
    /// Connect to the overlay's listening socket. ``path`` defaults to
    /// [`crate::protocol::SOCKET_PATH`]. The socket is set non-blocking
    /// so a slow / missing overlay doesn't stall our Qt event loop.
    pub fn connect<P: AsRef<Path>>(path: P) -> Result<Self> {
        let bytes = path.as_ref()
            .to_str()
            .and_then(|s| if s.as_bytes().len() < 108 { Some(s) } else { None })
            .ok_or(Error::InvalidArg("socket path too long or non-UTF8"))?
            .as_bytes();

        // Build sockaddr_un by zero-initialising then filling sun_path.
        let mut addr: libc::sockaddr_un = unsafe { mem::zeroed() };
        addr.sun_family = libc::AF_UNIX as libc::sa_family_t;
        for (dst, &src) in addr.sun_path.iter_mut().zip(bytes.iter()) {
            *dst = src as libc::c_char;
        }
        let addr_len = (mem::size_of::<libc::sa_family_t>() + bytes.len() + 1)
            as libc::socklen_t;

        // SOCK_SEQPACKET | SOCK_CLOEXEC | SOCK_NONBLOCK. CLOEXEC on
        // the creation call so fork+exec by the Qt process doesn't
        // leak the fd. NONBLOCK matches the "drop frame on overlay
        // backpressure" policy in `send`.
        let fd = unsafe {
            libc::socket(
                libc::AF_UNIX,
                libc::SOCK_SEQPACKET | libc::SOCK_CLOEXEC | libc::SOCK_NONBLOCK,
                0,
            )
        };
        if fd < 0 {
            return Err(Error::SocketIo(io::Error::last_os_error()));
        }
        let owned = unsafe { OwnedFd::from_raw_fd(fd) };

        let rc = unsafe {
            libc::connect(
                owned.as_raw_fd(),
                &addr as *const _ as *const libc::sockaddr,
                addr_len,
            )
        };
        if rc < 0 {
            return Err(Error::SocketIo(io::Error::last_os_error()));
        }

        Ok(Self { fd: owned })
    }

    /// Send a frame with an optional attached fd. On EAGAIN we drop
    /// the message (the overlay is behind; we'd rather skip a frame
    /// than accumulate latency). Returns Ok(false) on drop, Ok(true)
    /// on successful send.
    pub fn send(&self, frame: &Frame, fd: Option<RawFd>) -> Result<bool> {
        let payload = frame.as_bytes();

        let iov = libc::iovec {
            iov_base: payload.as_ptr() as *mut _,
            iov_len: payload.len(),
        };

        // Control message buffer (cmsghdr + one fd). Pre-allocated so
        // we don't touch the heap in the hot path.
        let mut cmsg_buf = [0u8; CMSG_BUFFER_SIZE];
        let (cmsg_ptr, cmsg_len) = match fd {
            Some(fd) => {
                let ptr = cmsg_buf.as_mut_ptr() as *mut libc::cmsghdr;
                unsafe {
                    (*ptr).cmsg_level = libc::SOL_SOCKET;
                    (*ptr).cmsg_type = libc::SCM_RIGHTS;
                    (*ptr).cmsg_len = libc_cmsg_len(mem::size_of::<RawFd>());
                    let data_ptr = libc_cmsg_data(ptr) as *mut RawFd;
                    std::ptr::write(data_ptr, fd);
                }
                (cmsg_buf.as_mut_ptr() as *mut _,
                 libc_cmsg_space(mem::size_of::<RawFd>()))
            }
            None => (std::ptr::null_mut(), 0),
        };

        let msg = libc::msghdr {
            msg_name: std::ptr::null_mut(),
            msg_namelen: 0,
            msg_iov: &iov as *const _ as *mut _,
            msg_iovlen: 1,
            msg_control: cmsg_ptr,
            msg_controllen: cmsg_len,
            msg_flags: 0,
        };

        let n = unsafe {
            libc::sendmsg(self.fd.as_raw_fd(), &msg, libc::MSG_NOSIGNAL)
        };
        if n >= 0 {
            return Ok(true);
        }
        let err = io::Error::last_os_error();
        // EAGAIN / EWOULDBLOCK: overlay hasn't drained; drop.
        if let Some(errno) = err.raw_os_error() {
            if errno == libc::EAGAIN || errno == libc::EWOULDBLOCK {
                return Ok(false);
            }
        }
        Err(Error::SocketIo(err))
    }
}

// ── cmsghdr helpers ──
// libc doesn't expose CMSG_*() macros directly from Rust; we re-implement
// them here. The kernel defines them for all POSIX-likes as:
//
//   CMSG_ALIGN(n) = round up to sizeof(size_t)
//   CMSG_LEN(n)   = CMSG_ALIGN(sizeof(struct cmsghdr)) + n
//   CMSG_SPACE(n) = CMSG_ALIGN(sizeof(struct cmsghdr)) + CMSG_ALIGN(n)
//   CMSG_DATA(c)  = (unsigned char *)((struct cmsghdr *)c + 1)

const fn cmsg_align(n: usize) -> usize {
    let a = mem::size_of::<usize>();
    (n + a - 1) & !(a - 1)
}

const fn libc_cmsg_len(n: usize) -> usize {
    cmsg_align(mem::size_of::<libc::cmsghdr>()) + n
}

const fn libc_cmsg_space(n: usize) -> usize {
    cmsg_align(mem::size_of::<libc::cmsghdr>()) + cmsg_align(n)
}

fn libc_cmsg_data(cmsg: *mut libc::cmsghdr) -> *mut u8 {
    unsafe { (cmsg as *mut u8).add(cmsg_align(mem::size_of::<libc::cmsghdr>())) }
}

// Buffer large enough for one fd-bearing cmsg. Sized at compile time.
const CMSG_BUFFER_SIZE: usize = libc_cmsg_space(mem::size_of::<RawFd>());

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cmsg_size_fits_one_fd() {
        // sizeof(cmsghdr) is 16 on glibc x86_64; CMSG_ALIGN rounds
        // the u32 fd payload to 8; total CMSG_SPACE(sizeof(int)) = 24.
        // We don't pin the exact size (glibc vs musl differ) but the
        // buffer must be at least large enough for the kernel.
        assert!(CMSG_BUFFER_SIZE >= mem::size_of::<libc::cmsghdr>() + 4);
    }

    #[test]
    fn send_to_closed_target_yields_socket_error() {
        // Connect to a nonexistent path: connect() itself fails.
        let res = FrameSocket::connect("/tmp/nonexistent-vsrg-test-sock");
        assert!(res.is_err(), "expected connect to unbound path to fail");
    }

    /// Spawn a ``socketpair(AF_UNIX, SOCK_SEQPACKET)`` and return
    /// both halves wrapped so the test can drive the producer API
    /// through ``a`` and read raw bytes via recvmsg on ``b``. This
    /// matches what production connects to (the gl_layer bind()s a
    /// SEQPACKET server), so behaviour is the same as end-to-end.
    fn socketpair_seqpacket() -> (OwnedFd, OwnedFd) {
        let mut fds = [0i32; 2];
        let rc = unsafe {
            libc::socketpair(libc::AF_UNIX, libc::SOCK_SEQPACKET, 0,
                             fds.as_mut_ptr())
        };
        assert!(rc == 0, "socketpair failed: errno={}", unsafe { *libc::__errno_location() });
        (
            unsafe { OwnedFd::from_raw_fd(fds[0]) },
            unsafe { OwnedFd::from_raw_fd(fds[1]) },
        )
    }

    #[test]
    fn roundtrip_publish_over_seqpacket_pair() {
        // Socketpair = producer/consumer in one process, so we can
        // validate framing + SCM_RIGHTS fd delivery without needing
        // EGL or a bound server.
        let (a, b) = socketpair_seqpacket();

        // The FrameSocket API expects a connected socket -- here we
        // cheat by constructing it directly around a matching fd.
        // Production always goes through ``connect()``.
        let prod = FrameSocket { fd: a };

        let frame = Frame::new_publish(7, 3, 640, 480,
                                       crate::protocol::FORMAT_ARGB8888, 0);

        // Use stdout fd 1 as a cheap "valid fd" to pass.
        let sent = prod.send(&frame, Some(1)).unwrap();
        assert!(sent, "send should succeed over a paired socket");

        // Receive on the other end via raw recvmsg so we also see the
        // SCM_RIGHTS ancillary data landed.
        let mut buf = [0u8; 128];
        let iov = libc::iovec {
            iov_base: buf.as_mut_ptr() as *mut _,
            iov_len: buf.len(),
        };
        let mut cmsg_buf = [0u8; 64];
        let mut msg: libc::msghdr = unsafe { std::mem::zeroed() };
        msg.msg_iov = &iov as *const _ as *mut _;
        msg.msg_iovlen = 1;
        msg.msg_control = cmsg_buf.as_mut_ptr() as *mut _;
        msg.msg_controllen = cmsg_buf.len();
        let n = unsafe { libc::recvmsg(b.as_raw_fd(), &mut msg, 0) };
        assert!(n > 0, "recvmsg failed");
        assert_eq!(n as usize, std::mem::size_of::<Frame>());

        let recv_frame: &Frame = unsafe { &*(buf.as_ptr() as *const Frame) };
        assert_eq!(recv_frame.magic, crate::protocol::MAGIC);
        assert_eq!(recv_frame.channel_id, 7);
        assert_eq!(recv_frame.generation, 3);

        // And: the fd landed as ancillary data.
        assert!(msg.msg_controllen > 0,
                "expected SCM_RIGHTS ancillary data");
    }
}
