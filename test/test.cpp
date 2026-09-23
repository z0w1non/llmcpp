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
    commandline_args(const std::vector<std::string>& args)
        : args_{ args }
    {
        build_argv();
    }

    commandline_args(std::initializer_list<std::string> args)
        : args_{ args.begin(), args.end()}
    {
        build_argv();
    }

    commandline_args(const commandline_args&) = delete;
    commandline_args& operator=(const commandline_args&) = delete;
    commandline_args(commandline_args&&) = delete;
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

struct scoped_nowide_cout_redirect
{
    scoped_nowide_cout_redirect()
        : old_buf_(boost::nowide::cout.rdbuf(buffer_.rdbuf()))
    {}

    ~scoped_nowide_cout_redirect()
    {
        boost::nowide::cout.rdbuf(old_buf_);
    }

    std::string str() const
    {
        return buffer_.str();
    }

private:
    std::ostringstream buffer_;
    std::streambuf* old_buf_;
};

TEST(test_exception_safe_main, invalid_option)
{
    commandline_args args{ "llmcpp.exe", "--invalid-option"};

    scoped_nowide_cout_redirect cout;
    EXPECT_NO_THROW(llmcpp::exception_safe_main(args.argc(), args.argv()));
    const std::string stdout_output{ cout.str() };

    EXPECT_THAT(stdout_output, testing::HasSubstr("[error]"));
}

TEST(test_exception_safe_main, help)
{
    commandline_args args{ "llmcpp.exe", "--help" };

    scoped_nowide_cout_redirect cout;
    EXPECT_NO_THROW(llmcpp::exception_safe_main(args.argc(), args.argv()));
    const std::string stdout_output{ cout.str() };

    EXPECT_THAT(stdout_output, testing::HasSubstr("Allowed options"));
}
