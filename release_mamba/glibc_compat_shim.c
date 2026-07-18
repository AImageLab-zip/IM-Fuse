/* Some GCC builds (e.g. Red Hat gcc-toolset SCL packages, as shipped in
 * pytorch/manylinux2_28-builder) link libstdc++'s atomic fast-path against
 * __libc_single_threaded, a symbol only present starting glibc 2.32 -
 * regardless of the actual glibc baseline being targeted. Providing our own
 * strong definition here satisfies that reference at link time, so the
 * produced .so no longer needs glibc >= 2.32 at runtime (it just always
 * takes the thread-safe path instead of the single-threaded fast path). */
int __libc_single_threaded = 0;
