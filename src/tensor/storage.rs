use std::marker::PhantomData;
use std::alloc::{alloc, dealloc, Layout};

// This will represents the device where the tensor is stored
pub enum Device {
    CPU,
    Metal
}

// This will represents the storage of the tensor
pub struct Storage<T> {
    ptr: *mut T,        // Raw pointer to our actual data in memory
    len: usize,         // Total number of elements we can store
    device: Device,     // Where the data lives (CPU/GPU/etc)
    _marker: PhantomData<T>, // Type information for Rust's type system
}

// Implementating the creation of a new storage
impl<T> Storage<T> {
    pub fn new(size: usize, device: Device) -> Self {
        let (ptr, len) = match device {
            Device::CPU => {
                // Existing CPU allocation
                let layout = Layout::array::<T>(size).unwrap();
                let ptr = unsafe { alloc(layout) as *mut T };
                (ptr, size)
            },
            Device::Metal => {
                // TODO: Implement Metal allocation
                // For now, fallback to CPU
                let layout = Layout::array::<T>(size).unwrap();
                let ptr = unsafe { alloc(layout) as *mut T };
                (ptr, size)
            }
        };

        Self {
            ptr,
            len,
            device,
            _marker: PhantomData,
        }
    }
    
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn as_slice(&self) -> &[T] {
        unsafe {
            std::slice::from_raw_parts(self.ptr, self.len)
        }
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe {
            std::slice::from_raw_parts_mut(self.ptr, self.len)
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_storage_creation() {
        let storage: Storage<f32> = Storage::new(10, Device::CPU);
        assert_eq!(storage.len(), 10);
    }

    #[test]
    fn test_storage_slice_access() {
        let mut storage: Storage<i32> = Storage::new(5, Device::CPU);
        
        // Test mutable slice
        let slice = storage.as_mut_slice();
        slice[0] = 42;
        slice[4] = 24;

        // Test immutable slice
        let slice = storage.as_slice();
        assert_eq!(slice[0], 42);
        assert_eq!(slice[4], 24);
        assert_eq!(slice.len(), 5);
    }

    #[test]
    fn test_zero_sized_storage() {
        let storage: Storage<f64> = Storage::new(0, Device::CPU);
        assert_eq!(storage.len(), 0);
        assert_eq!(storage.as_slice().len(), 0);
    }
}
