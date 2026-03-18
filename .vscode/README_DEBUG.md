# Debugging Guide

This project is now configured for debugging with breakpoints in VS Code.

## How to Debug:

### Method 1: Using VS Code Debug Panel (Recommended)

1. **Set Breakpoints**: Click in the left margin of any line in your code files (`.cpp` or `.h`) to set a breakpoint (red dot will appear)

2. **Start Debugging**:
   - Press `F5` OR
   - Go to "Run and Debug" panel (Ctrl+Shift+D)
   - Select "C++ Debug" from the dropdown
   - Click the green play button

3. **Debug Controls**:
   - `F5` - Continue
   - `F10` - Step Over
   - `F11` - Step Into
   - `Shift+F11` - Step Out
   - `Shift+F5` - Stop

4. **View Variables**: 
   - Hover over variables to see their values
   - Check the "Variables" panel in the sidebar
   - Add variables to "Watch" panel for monitoring

### Method 2: Using GDB from Terminal

```bash
cd /sfa3d_adcm/build
gdb ./HailoPerceptionProject

# Inside GDB:
(gdb) break main          # Set breakpoint at main()
(gdb) break filename.cpp:line_number  # Set breakpoint at specific line
(gdb) run                 # Start debugging
(gdb) next                # Step over (like F10)
(gdb) step                # Step into (like F11)
(gdb) print variable_name # Print variable value
(gdb) continue            # Continue execution
(gdb) quit                # Exit GDB
```

### Building for Debug:

- The project is already configured to build in Debug mode by default
- To rebuild in Debug: Press `Ctrl+Shift+B` and select "build"
- Or run: `cd build && cmake .. -DCMAKE_BUILD_TYPE=Debug && make`

## Notes:

- Debug symbols are included (you can see source code line numbers)
- If you need to rebuild, the debug build will maintain debug symbols
- Make sure you have the `gdb` debugger installed: `sudo apt-get install gdb`

