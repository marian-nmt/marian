#include "marian.h"
#include "translator/beam_search.h"
#include "translator/translator.h"

#include "3rd_party/simple-websocket-server/server_ws.hpp"

typedef SimpleWeb::SocketServer<SimpleWeb::WS> WsServer;

int main(int argc, char** argv) {
  using namespace marian;

  auto options = New<Config>(argc, argv, ConfigMode::translating);
  auto task = New<TranslateLoopMultiGPU<BeamSearch>>(options);

  WsServer server_;
  server_.config.port = 1234;

  auto &echo = server_.endpoint["^/translate/?$"];

  echo.on_message = [&task](std::shared_ptr<WsServer::Connection> connection,
                       std::shared_ptr<WsServer::Message> message) {
    auto message_str = message->string();

    std::cout << "Server: Message received: \"" << message_str << "\" from "
              << connection.get() << std::endl;
    std::cout << "Server: Sending message \"" << message_str << "\" to "
              << connection.get() << std::endl;

    auto send_stream = std::make_shared<WsServer::SendStream>();
    boost::timer::cpu_timer timer;
    for(auto& transl : task->run({message_str})) {
      *send_stream << transl << std::endl;
    }
    LOG(info)->info("Search took: {}", timer.format(5, "%ws"));

    // connection->send is an asynchronous function
    connection->send(send_stream, [](const SimpleWeb::error_code &ec) {
      if(ec) {
        std::cout << "Server: Error sending message. " <<
            // Error Codes for error code meanings:
            // http://www.boost.org/doc/libs/1_55_0/doc/html/boost_asio/reference.html
            "Error: " << ec << ", error message: " << ec.message() << std::endl;
      }
    });

    // Alternatively, using a convenience function:
    // connection->send(message_str, [](const SimpleWeb::error_code & /*ec*/) {
    // /*handle error*/ });
  };

  echo.on_open = [](std::shared_ptr<WsServer::Connection> connection) {
    std::cout << "Server: Opened connection " << connection.get() << std::endl;
  };

  // See RFC 6455 7.4.1. for status codes
  echo.on_close = [](std::shared_ptr<WsServer::Connection> connection,
                     int status,
                     const std::string & /*reason*/) {
    std::cout << "Server: Closed connection " << connection.get()
              << " with status code " << status << std::endl;
  };

  // Error Codes for error code meanings
  // http://www.boost.org/doc/libs/1_55_0/doc/html/boost_asio/reference.html,
  echo.on_error = [](std::shared_ptr<WsServer::Connection> connection,
                     const SimpleWeb::error_code &ec) {
    std::cout << "Server: Error in connection " << connection.get() << ". "
              << "Error: " << ec << ", error message: " << ec.message()
              << std::endl;
  };

  std::thread server_thread([&server_]() {
    LOG(info)->info("Server started");
    server_.start();
  });

  server_thread.join();

  return 0;
}
