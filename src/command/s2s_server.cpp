#include "marian.h"
#include "translator/beam_search.h"
#include "translator/translator.h"

#include "3rd_party/simple-websocket-server/server_ws.hpp"

typedef SimpleWeb::SocketServer<SimpleWeb::WS> WsServer;

int main(int argc, char **argv) {
  using namespace marian;

  auto options = New<Config>(argc, argv, ConfigMode::translating);
  auto task = New<TranslateLoopMultiGPU<BeamSearch>>(options);

  WsServer server_;
  server_.config.port = 1234;

  auto &translate = server_.endpoint["^/translate/?$"];

  translate.on_message = [&task](Ptr<WsServer::Connection> connection,
                                 Ptr<WsServer::Message> message) {
    auto message_str = message->string();
    LOG(info)->info("Message received: \"" + message_str + "\"");

    auto send_stream = std::make_shared<WsServer::SendStream>();
    boost::timer::cpu_timer timer;
    for(auto &transl : task->run({message_str})) {
      *send_stream << transl << std::endl;
    }
    LOG(info)->info("Search took: {}", timer.format(5, "%ws"));

    connection->send(send_stream, [](const SimpleWeb::error_code &ec) {
      if(ec) {
        auto ec_str = std::to_string(ec.value());
        LOG(warn)
            ->warn("Error sending message: (" + ec_str + ") " + ec.message());
      }
    });
  };

  // Error Codes for error code meanings
  // http://www.boost.org/doc/libs/1_55_0/doc/html/boost_asio/reference.html
  translate.on_error = [](Ptr<WsServer::Connection> connection,
                          const SimpleWeb::error_code &ec) {
    auto ec_str = std::to_string(ec.value());
    LOG(warn)->warn("Connection error: (" + ec_str + ") " + ec.message());
  };

  std::thread server_thread([&server_]() {
    LOG(info)->info("Server started");
    server_.start();
  });

  server_thread.join();

  return 0;
}
