#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "llmcpp.hpp"

#include <vector>
#include <initializer_list>

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

struct commandline_args
{
    explicit commandline_args(const std::vector<std::string>& args)
        : args_{ args }
    {
        build_argv();
    }

    commandline_args(std::initializer_list<std::string> args)
        : args_{ args.begin(), args.end() }
    {
        build_argv();
    }

    commandline_args(const commandline_args&) = delete;
    commandline_args(commandline_args&&) = delete;
    commandline_args& operator=(const commandline_args&) = delete;
    commandline_args& operator=(commandline_args&&) = delete;

    int argc() const
    {
        return static_cast<int>(args_.size());
    }

    char** argv()
    {
        return argv_.data();
    }

private:
    void build_argv()
    {
        argv_.clear();
        argv_.reserve(args_.size() + 1);
        for (std::string& arg : args_)
        {
            argv_.push_back(arg.data());
        }
        argv_.push_back(nullptr);
    }

    std::vector<std::string> args_;
    std::vector<char*> argv_;
};

struct scoped_ostream_redirect
{
    scoped_ostream_redirect(std::ostream& ostream)
        : ostream_{ ostream }
        , old_buf_{ ostream_.rdbuf(buffer_.rdbuf()) }
    {
    }

    ~scoped_ostream_redirect()
    {
        ostream_.rdbuf(old_buf_);
    }

    scoped_ostream_redirect(const scoped_ostream_redirect&) = delete;
    scoped_ostream_redirect(scoped_ostream_redirect&&) = delete;
    scoped_ostream_redirect& operator=(const scoped_ostream_redirect&) = delete;
    scoped_ostream_redirect& operator=(scoped_ostream_redirect&&) = delete;

    std::string str() const
    {
        return buffer_.str();
    }

private:
    std::ostream& ostream_;
    std::ostringstream buffer_;
    std::streambuf* old_buf_;
};

struct help_option_test : testing::TestWithParam<const char*> {};

TEST_P(help_option_test, help)
{
    const char* option{ GetParam() };
    commandline_args args{ "llmcpp.exe", option };
    const scoped_ostream_redirect cout{ boost::nowide::cout };

    llmcpp::command_line::parse_result result{};
    llmcpp::config cfg;
    EXPECT_NO_THROW({
        result = llmcpp::command_line::parse(args.argc(), args.argv(), cfg);
        });
    EXPECT_EQ(result, llmcpp::command_line::parse_result::help);

    const std::string cout_output{ cout.str() };
    EXPECT_THAT(cout_output, testing::HasSubstr("Allowed options"));
}

INSTANTIATE_TEST_SUITE_P(
    parse,
    help_option_test,
    testing::Values("--help", "-h")
);

TEST(parse, unrecognised_options)
{
    commandline_args args{ "llmcpp.exe", "--unrecognised-option" };
    const scoped_ostream_redirect cerr{ boost::nowide::cerr };

    llmcpp::command_line::parse_result result{};
    llmcpp::config cfg;
    EXPECT_NO_THROW({
        result = llmcpp::command_line::parse(args.argc(), args.argv(), cfg);
        });
    EXPECT_EQ(result, llmcpp::command_line::parse_result::program_options_error);

    const std::string cerr_output{ cerr.str() };
    EXPECT_THAT(cerr_output, testing::HasSubstr("unrecognised option"));
    EXPECT_THAT(cerr_output, testing::HasSubstr("--unrecognised-option"));
}

TEST(test_exception_safe_main, unrecognised_option)
{
    commandline_args args{ "llmcpp.exe", "--unrecognised-option" };
    const scoped_ostream_redirect cerr{ boost::nowide::cerr };

    int result{};
    EXPECT_NO_THROW({
        result = llmcpp::exception_safe_main(args.argc(), args.argv());
        });
    EXPECT_NE(result, 0);
}

TEST(exception_safe_main, help)
{
    commandline_args args{ "llmcpp.exe", "--help" };
    const scoped_ostream_redirect cout{ boost::nowide::cout };

    int result{};
    EXPECT_NO_THROW({
        result = llmcpp::exception_safe_main(args.argc(), args.argv());
        });
    EXPECT_EQ(result, 0);
}

TEST(exception_safe_main, help_short)
{
    commandline_args args{ "llmcpp.exe", "-h" };
    const scoped_ostream_redirect cout{ boost::nowide::cout };

    int result{};
    EXPECT_NO_THROW({
        result = llmcpp::exception_safe_main(args.argc(), args.argv());
        });
    EXPECT_EQ(result, 0);
}

TEST(lru_cache, get_tokens)
{
    using cache_type = llmcpp::lru_cache<1>;
    constexpr int factor{ 2 };
    auto callback{ [](std::string_view str) -> int { return static_cast<int>(str.size()) * factor; } };

    cache_type cache{ callback };
    const std::string str{ "test" };
    EXPECT_EQ(cache.get_tokens(str), callback(str));
}

TEST(lru_cache, lru)
{
    using cache_type = llmcpp::lru_cache<3>;
    int count{};
    auto callback{ [&count](std::string_view str) mutable -> int { return count++; } };

    cache_type cache{ callback };
    EXPECT_EQ(cache.get_tokens("a"), 0);
    EXPECT_EQ(cache.get_tokens("b"), 1);
    EXPECT_EQ(cache.get_tokens("a"), 0);
    EXPECT_EQ(cache.get_tokens("b"), 1);
    EXPECT_EQ(cache.get_tokens("c"), 2);
    EXPECT_EQ(cache.get_tokens("a"), 0);
    EXPECT_EQ(cache.get_tokens("b"), 1);
    EXPECT_EQ(cache.get_tokens("c"), 2);
    EXPECT_EQ(cache.get_tokens("d"), 3);
    EXPECT_EQ(cache.get_tokens("a"), 4);
}
