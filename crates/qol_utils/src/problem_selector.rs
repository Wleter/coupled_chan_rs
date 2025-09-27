use std::collections::VecDeque;

use anyhow::Result;

/// Gets the arguments from the command line and returns them as a VecDeque
pub fn get_args() -> VecDeque<String> {
    let mut args = std::env::args();
    // get rid of the first argument which is the program name
    args.next();

    args.collect()
}

pub type ProblemMethod = Box<dyn Fn(&mut VecDeque<String>) -> Result<()>>;

/// Trait for selecting a problem to run
pub trait ProblemSelector {
    /// Name of the problem.
    const NAME: &'static str;

    /// Vector of all available problems to choose
    fn list() -> Vec<(&'static str, ProblemMethod)>;

    /// Select a problem to run preselected or from user input
    /// The problem can be run with -1 to run all problems.
    fn select(args: &mut VecDeque<String>) {
        let arg = args.pop_front();

        match arg {
            Some(arg) => {
                select(&arg, args, &Self::list());
            }
            None => {
                println!("");
                println!("Provide a problem number:");
                println!("-1: run all problems");

                let problems = Self::list();
                for (i, (problem, _)) in problems.iter().enumerate() {
                    println!("{i}: {problem}");
                }

                let mut input = String::new();
                std::io::stdin().read_line(&mut input).unwrap();
                let input = input.trim();

                select(input, args, &Self::list());
            }
        }
    }
}

fn select(arg: &str, args: &mut VecDeque<String>, method_list: &[(&'static str, ProblemMethod)]) {
    if arg == "-1" {
        let mut args = VecDeque::from(vec!["-1".to_string()]);

        for (name, method) in method_list {
            println!("Selected: {name}");
            let result = method(&mut args);

            if let Err(err) = result {
                println!("{err}")
            }
        }
    } else {
        let arg = arg.parse::<usize>().expect("Selected problem should be a number");
        assert!(arg < method_list.len(), "Selected problem number exceed problem list");

        println!("Selected: {}", method_list[arg].0);
        let result = method_list[arg].1(args);

        if let Err(err) = result {
            println!("{err}")
        }
    }
}

#[macro_export]
/// Implement ProblemSelector trait
///
/// # Syntax
///
/// - `problems_impl!(selector, name, (problem_name => method),*)`
macro_rules! problems_impl {
    ($selector:ty, $name:expr, $($problem_name:expr => $method:expr),* $(,)?) => {
        impl $crate::problem_selector::ProblemSelector for $selector {
            const NAME: &'static str = $name;

            fn list() -> Vec<(&'static str, $crate::problem_selector::ProblemMethod)> {
                vec![$(($problem_name, Box::new($method))),*]
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use std::{collections::VecDeque, sync::Mutex};

    use crate::problem_selector::ProblemSelector;

    struct TestProblems;

    static CALLED: Mutex<bool> = Mutex::new(false);

    problems_impl!(TestProblems, "test",
        "test1" => |_| Ok(println!("test1")),
        "test2" => |_| Ok(println!("test2")),
        "test3" => |args| {
            *CALLED.lock().unwrap() = true;
            println!("test3, args = {args:?}");

            Err(anyhow::anyhow!("error"))
        }
    );

    #[test]
    fn test_problem_selector() {
        assert!(!*CALLED.lock().unwrap());
        TestProblems::select(&mut VecDeque::from_iter(["-1".to_string()]));
        assert!(*CALLED.lock().unwrap());
    }
}
