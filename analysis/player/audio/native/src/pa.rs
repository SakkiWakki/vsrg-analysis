//! Minimal PortAudio FFI. We link the system ``libportaudio`` (the same shared
//! library sounddevice loads) and declare only what the ring-draining output
//! stream needs. May need to extend when deprecating the Py fallback

use std::os::raw::{c_double, c_int, c_ulong, c_void};

pub type PaError = c_int;
pub type PaDeviceIndex = c_int;
pub type PaSampleFormat = c_ulong;
pub type PaStream = c_void;
pub type PaStreamCallbackFlags = c_ulong;
pub type PaTime = c_double;

pub const PA_NO_ERROR: PaError = 0;
pub const PA_FLOAT32: PaSampleFormat = 0x0000_0001;
/// Return value from the callback: keep the stream running.
pub const PA_CONTINUE: c_int = 0;

#[repr(C)]
pub struct PaStreamParameters {
    pub device: PaDeviceIndex,
    pub channel_count: c_int,
    pub sample_format: PaSampleFormat,
    pub suggested_latency: PaTime,
    pub host_api_specific_stream_info: *mut c_void,
}

#[repr(C)]
pub struct PaStreamCallbackTimeInfo {
    pub input_buffer_adc_time: PaTime,
    pub current_time: PaTime,
    pub output_buffer_dac_time: PaTime,
}

/// Signature PortAudio calls on its real-time thread. `user_data` is our
/// `*const StreamCtx`. MUST NOT acquire the GIL or allocate.
pub type PaStreamCallback = unsafe extern "C" fn(
    input: *const c_void,
    output: *mut c_void,
    frame_count: c_ulong,
    time_info: *const PaStreamCallbackTimeInfo,
    status_flags: PaStreamCallbackFlags,
    user_data: *mut c_void,
) -> c_int;

#[link(name = "portaudio")]
extern "C" {
    pub fn Pa_Initialize() -> PaError;
    pub fn Pa_Terminate() -> PaError;
    pub fn Pa_GetDefaultOutputDevice() -> PaDeviceIndex;
    pub fn Pa_OpenStream(
        stream: *mut *mut PaStream,
        input_parameters: *const PaStreamParameters,
        output_parameters: *const PaStreamParameters,
        sample_rate: c_double,
        frames_per_buffer: c_ulong,
        stream_flags: c_ulong,
        stream_callback: Option<PaStreamCallback>,
        user_data: *mut c_void,
    ) -> PaError;
    pub fn Pa_StartStream(stream: *mut PaStream) -> PaError;
    pub fn Pa_StopStream(stream: *mut PaStream) -> PaError;
    pub fn Pa_CloseStream(stream: *mut PaStream) -> PaError;
    pub fn Pa_GetStreamTime(stream: *mut PaStream) -> PaTime;
}
