`storage.rs`

The Storage structure serves as the memory management layer of our tensor system. The key purposes of Storage are:

- Memory abstraction

```rust
impl<T> Storage<T> {
    pub fn new(size: usize) -> Self {
        unsafe {
            let layout = Layout::array::<T>(size).unwrap();
            let ptr = alloc(layout) as *mut T;  // Allocate raw memory
            
            Self { /* ... */ }
        }
    }
}
```
This provides a safe interface to manage raw memory. Users of Storage don't need to worry about memory allocation/deallocation.

- Device management

```rust
pub enum Device {
    CPU,
    CUDA(i32),
    Metal,
}
```
Storage handles where the data lives - it could be in CPU memory, GPU memory, or other accelerators.

`tensor.rs`
