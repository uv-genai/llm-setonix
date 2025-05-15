#include <coroutine>
#include <iostream>
#include <memory>

// A simple coroutine that yields values sequentially
struct Generator {
    struct promise_type {
        int current_value;

        Generator get_return_object() { return Generator{this}; }
        auto initial_suspend() { return std::suspend_always{}; }
        auto final_suspend() noexcept { return std::suspend_always{}; }
        void return_void() {}
        void unhandled_exception() { std::terminate(); }
        auto yield_value(int value) {
            current_value = value;
            return std::suspend_always{};
        }
    };

    using handle_type = std::coroutine_handle<promise_type>;
    handle_type coro_handle;

    Generator(promise_type* p) : coro_handle(handle_type::from_promise(*p)) {}
    ~Generator() { if (coro_handle) coro_handle.destroy(); }

    int next() {
        coro_handle.resume();
        return coro_handle.promise().current_value;
    }
};

Generator range(int start, int end) {
    for (int i = start; i <= end; ++i) {
        co_yield i;  // Pauses execution and yields a value
    }
}

int main() {
    auto gen = range(1, 5);
    
    while (auto val = gen.next()) {
        std::cout << "Received: " << val << "\n";
    }

    return 0;
}
