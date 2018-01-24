    
    #include <iostream>
    #include <memory>
    #include <fstream>
    
    #include <boost/timer/timer.hpp>
    
    #include <SQLiteCpp/SQLiteCpp.h>
    
    int main(int argc, char** argv) {
        
        SQLite::Database db("corpus.db", SQLite::OPEN_READWRITE|SQLite::OPEN_CREATE);
        db.exec("PRAGMA temp_store_directory = '/data1/marcinjd';");
        
        db.exec("drop table if exists lines");    
        db.exec("create table lines (_id integer, line0 text, line1 text);");
        
        boost::timer::auto_cpu_timer total;
        
        std::unique_ptr<boost::timer::auto_cpu_timer> t(new boost::timer::auto_cpu_timer());
           
        SQLite::Statement ps(db, "insert into lines values (?, ?, ?)");
           
        std::string line0, line1;
        size_t lines = 0;
        
        std::cerr << "Reading from " << argv[1] << " and " << argv[2] << std::endl;

        std::ifstream file0(argv[1]);
        std::ifstream file1(argv[2]);
                
        db.exec("begin;");
        while(std::getline(file0, line0) && std::getline(file1, line1)) {
            ps.bind(1, (int)lines);
            ps.bind(2, line0);
            ps.bind(3, line1);
            
            ps.exec();
            ps.reset();
            
            lines++;
            if(lines % 1000000 == 0) {
                std::cerr << "[" << lines << "]" << std::endl;
                t.reset(new boost::timer::auto_cpu_timer());
                
                db.exec("commit;");
                db.exec("begin;");
            }
        }
        db.exec("commit;");
        
        std::cerr << "[" << lines << "]" << std::endl;
        
        t.reset(new boost::timer::auto_cpu_timer());
        std::cerr << "creating index" << std::endl;    
        db.exec("create unique index idx_line on lines (_id);");
        
        t.reset(new boost::timer::auto_cpu_timer());

        std::cout << "count : " << db.execAndGet("select count(*) from lines").getInt() << std::endl;        
        t.reset(new boost::timer::auto_cpu_timer());
        
        int count = 0;
        SQLite::Statement sel(db, "select * from lines order by random();");
        t.reset(new boost::timer::auto_cpu_timer());
        while(sel.executeStep()) {
            // Demonstrate how to get some typed column value
            int id = sel.getColumn(0);
            std::string value0 = sel.getColumn(1);
            std::string value1 = sel.getColumn(2);
            
            if(count % 1000000 == 0)
                std::cout << count << " " << id << "\t" << value0 << "\t" << value1 << std::endl;
            count++;
        }

        return 0;
    }