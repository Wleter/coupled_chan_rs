pub mod params;
pub mod problem_selector;

/// Macro for crating cache with given name `$cache_name:ident` around the expression
/// It is used together with `cached_mel!` macro to cache subsequent calculations.
///
/// # Syntax
///
/// - `make_cache!($cache_name:ident, $body:expr)`
/// - `make_cache!($cache_name:ident => $capacity:expr, $body:expr)`
///
/// # Arguments
///
/// ## General Arguments
/// - `$cache_name`: Name of the create cache.
/// - `$body`: body around which to create the cache.
///
/// ## Optional Arguments
/// - `$capacity` (optional): Initial capacity of the cache.
#[macro_export]
macro_rules! make_cache {
    ($cache_name:ident => $capacity:expr, $body:expr) => {{
        let mut $cache_name = std::collections::HashMap::with_capacity($capacity);
        $body
    }};
    ($cache_name:ident, $body:expr) => {{
        let mut $cache_name = std::collections::HashMap::new();
        $body
    }};
}

/// Macro for caching function inside cache_function! macro
///
/// # Syntax
///
/// - `cache_function!($cache_name:ident, $func:ident($($arg:expr),*))`
/// - `cache_function!($cache_name:ident, |[$($arg:ident),*]| $body:block)`
///
/// # Arguments
///
/// ## General Arguments
/// - `$cache_name`: Name of the used cache.
/// - `$func($($arg:ident),*)`: Function outputs to be cached with `arg` keys.
/// - or `|[$($arg:ident),*]| $body:block`: Closure to be cached
///
/// # Matched Arms
///
/// ## Arm 1: `cache_function!($cache_name, $func($($arg),*))`
/// Use this arm when you want to cache function
///
/// ## Arm 2: `cache_function!($cache_name, |[$($arg),*]| $body)`
/// Use this arm when you want to cache a closure in an operator
#[macro_export]
macro_rules! cache_function {
    ($cache_name:ident, $func:ident($($arg:expr),*)) => {{
        if let Some(result) = $cache_name.get(&($($arg),*)) {
            return *result;
        }

        let result = $func($($arg),*);
        $cache_name.insert(($($arg),*), result);
        result
    }};
    ($cache_name:ident, |[$($arg:ident),*]| $body:block) => {
        |[$($arg),*]| {
            if let Some(result) = $cache_name.get(&($($arg),*)) {
                return *result;
            }

            let result = $body;
            $cache_name.insert(($($arg),*), result);
            result
        }
    };
}

#[cfg(test)]
mod tests {
    use std::{
        thread::sleep,
        time::{Duration, Instant},
    };

    const DURATION: u64 = 100;

    fn long_computation<const N: usize>(a: [usize; N]) -> [usize; N] {
        sleep(Duration::from_millis(DURATION));
        a
    }

    fn print_computations<const N: usize>(a: Vec<[usize; N]>, mut f: impl FnMut([usize; N]) -> [usize; N]) -> Vec<f64> {
        a.iter()
            .map(|&a| {
                let start = Instant::now();
                let a = f(a);
                let end = start.elapsed();
                println!("result {:?} time: {}", a, end.as_secs_f64());

                end.as_secs_f64() * 1000.
            })
            .collect()
    }

    #[test]
    fn test_caching() {
        let values = vec![[3], [3], [4]];
        let durations = make_cache!(
            cache,
            print_computations(values, |a| { cache_function!(cache, long_computation(a)) })
        );
        assert!(durations[0] >= DURATION as f64);
        assert!(durations[1] < DURATION as f64);
        assert!(durations[2] >= DURATION as f64);

        let values = vec![[3], [3], [4]];
        let durations = make_cache!(
            cache,
            print_computations(values, cache_function!(cache, |[a]| { long_computation([a]) }))
        );
        assert!(durations[0] >= DURATION as f64);
        assert!(durations[1] < DURATION as f64);
        assert!(durations[2] >= DURATION as f64);
    }
}
