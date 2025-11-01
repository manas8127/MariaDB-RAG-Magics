-- OpenFlights Airport Data for MariaDB RAG Demo
-- Real-world aviation dataset for hackathon demonstration

USE rag_demo;

-- Create airports table for OpenFlights data
DROP TABLE IF EXISTS airports;
CREATE TABLE airports (
    id INT PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    city VARCHAR(255) NOT NULL,
    country VARCHAR(255) NOT NULL,
    iata_code VARCHAR(3),
    icao_code VARCHAR(4),
    latitude DECIMAL(10, 6),
    longitude DECIMAL(10, 6),
    altitude INT,
    timezone_offset DECIMAL(4, 2),
    dst VARCHAR(1),
    timezone VARCHAR(50),
    type VARCHAR(20),
    source VARCHAR(50),
    description TEXT,
    content_vector VECTOR(384) DEFAULT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    -- Indexes for better performance
    INDEX idx_city (city),
    INDEX idx_country (country),
    INDEX idx_iata (iata_code),
    INDEX idx_icao (icao_code),
    INDEX idx_type (type)
) ENGINE=InnoDB;

-- Enhanced OpenFlights sample data with rich descriptions for RAG demonstrations
INSERT INTO airports (id, name, city, country, iata_code, icao_code, latitude, longitude, altitude, timezone_offset, dst, timezone, type, source, description) VALUES 
(1, "Goroka Airport", "Goroka", "Papua New Guinea", "GKA", "AYGA", -6.081689, 145.391998, 5282, 10, "U", "Pacific/Port_Moresby", "airport", "OurAirports", "Goroka Airport serves the highland town of Goroka in Papua New Guinea. Located at an elevation of 5,282 feet, this regional airport is crucial for connecting the mountainous interior with major cities. The airport handles domestic flights and is vital for tourism to the Eastern Highlands Province, known for its coffee plantations and cultural festivals."),

(2, "Madang Airport", "Madang", "Papua New Guinea", "MAG", "AYMD", -5.207079, 145.789001, 20, 10, "U", "Pacific/Port_Moresby", "airport", "OurAirports", "Madang Airport is a coastal airport serving the beautiful town of Madang, known as the prettiest town in Papua New Guinea. At just 20 feet above sea level, it provides access to stunning coral reefs, tropical islands, and world-class diving sites. The airport is a gateway to adventure tourism and marine exploration in the Bismarck Sea."),

(3, "Mount Hagen Kagamuga Airport", "Mount Hagen", "Papua New Guinea", "HGU", "AYMH", -5.826789, 144.296005, 5388, 10, "U", "Pacific/Port_Moresby", "airport", "OurAirports", "Mount Hagen Kagamuga Airport is the primary airport for the Western Highlands Province of Papua New Guinea. At 5,388 feet elevation, it serves as a crucial hub for the coffee-growing region and provides access to traditional Highland cultures. The airport connects remote communities and supports agricultural exports, particularly high-quality Arabica coffee."),

(4, "Keflavik International Airport", "Keflavik", "Iceland", "KEF", "BIKF", 63.985000, -22.605600, 171, 0, "N", "Atlantic/Reykjavik", "airport", "OurAirports", "Keflavik International Airport is Iceland's main international gateway, located on the Reykjanes Peninsula. This modern airport serves as a major hub for transatlantic flights and is famous for its role in connecting Europe and North America. The airport offers stunning views of volcanic landscapes and is the primary entry point for tourists visiting Iceland's natural wonders including geysers, glaciers, and the Northern Lights."),

(5, "Heathrow Airport", "London", "United Kingdom", "LHR", "EGLL", 51.469603, -0.461941, 83, 0, "E", "Europe/London", "airport", "OurAirports", "London Heathrow Airport is one of the world's busiest international airports and serves as the primary hub for British Airways. Located west of Central London, Heathrow handles over 80 million passengers annually and connects to more international destinations than any other airport. The airport features state-of-the-art terminals, extensive shopping and dining, and serves as a major economic driver for the UK."),

(6, "John F. Kennedy International Airport", "New York", "United States", "JFK", "KJFK", 40.639751, -73.778924, 13, -5, "A", "America/New_York", "airport", "OurAirports", "John F. Kennedy International Airport is New York City's primary international airport, located in Queens. Named after the 35th President, JFK is a major hub for international travel to and from the United States. The airport features six terminals, handles over 60 million passengers annually, and serves as a gateway to the cultural and financial capital of America."),

(7, "Dubai International Airport", "Dubai", "United Arab Emirates", "DXB", "OMDB", 25.252777, 55.364444, 62, 4, "U", "Asia/Dubai", "airport", "OurAirports", "Dubai International Airport is a major aviation hub in the Middle East and one of the world's busiest airports by international passenger traffic. Known for its luxury shopping, world-class dining, and exceptional passenger experience, DXB serves as the flagship hub for Emirates Airlines. The airport connects six continents and is a crucial stopover point for long-haul international flights."),

(8, "Tokyo Haneda Airport", "Tokyo", "Japan", "HND", "RJTT", 35.552250, 139.779602, 35, 9, "U", "Asia/Tokyo", "airport", "OurAirports", "Tokyo Haneda Airport is one of two primary airports serving the Greater Tokyo Area and is closer to downtown Tokyo than Narita. Known for its exceptional efficiency, cleanliness, and customer service, Haneda primarily handles domestic flights and short-haul international routes. The airport showcases Japanese hospitality and technology, featuring automated systems and traditional Japanese design elements."),

(9, "Singapore Changi Airport", "Singapore", "Singapore", "SIN", "WSSS", 1.350189, 103.994433, 22, 8, "U", "Asia/Singapore", "airport", "OurAirports", "Singapore Changi Airport is consistently rated as one of the world's best airports, famous for its innovative design, lush gardens, and exceptional passenger amenities. The airport features the world's tallest indoor waterfall, butterfly garden, movie theaters, and even a swimming pool. Changi serves as a major hub for Southeast Asian travel and showcases Singapore's commitment to excellence in aviation."),

(10, "Los Angeles International Airport", "Los Angeles", "United States", "LAX", "KLAX", 33.942536, -118.408075, 125, -8, "A", "America/Los_Angeles", "airport", "OurAirports", "Los Angeles International Airport is the primary airport serving Los Angeles and is one of the busiest airports in the world. Known for its distinctive Theme Building and iconic LAX signage, the airport serves as a major gateway to the Pacific Rim and Latin America. LAX is closely associated with Hollywood glamour and serves millions of passengers traveling to the entertainment capital of the world."),

(11, "Charles de Gaulle Airport", "Paris", "France", "CDG", "LFPG", 49.012779, 2.550000, 392, 1, "E", "Europe/Paris", "airport", "OurAirports", "Charles de Gaulle Airport is the largest international airport in France and a major European hub. Named after the French president, CDG is known for its distinctive modernist architecture and serves as the primary hub for Air France. The airport connects Paris to destinations worldwide and serves as a gateway to European culture, fashion, and cuisine."),

(12, "Sydney Kingsford Smith Airport", "Sydney", "Australia", "SYD", "YSSY", -33.946609, 151.177002, 21, 10, "O", "Australia/Sydney", "airport", "OurAirports", "Sydney Kingsford Smith Airport is Australia's oldest and busiest airport, serving the vibrant city of Sydney. Located near Botany Bay, the airport offers stunning views of the Sydney Harbour and city skyline. SYD serves as the primary gateway to Australia and the Pacific, welcoming millions of visitors to explore the Sydney Opera House, Harbour Bridge, and beautiful beaches."),

(13, "Amsterdam Airport Schiphol", "Amsterdam", "Netherlands", "AMS", "EHAM", 52.308613, 4.763889, -11, 1, "E", "Europe/Amsterdam", "airport", "OurAirports", "Amsterdam Airport Schiphol is the main international airport of the Netherlands and one of Europe's busiest airports. Built on reclaimed land below sea level, Schiphol is known for its innovative design, art exhibitions, and excellent connectivity. The airport serves as the hub for KLM Royal Dutch Airlines and is a major European gateway known for its efficiency and passenger-friendly facilities."),

(14, "Frankfurt Airport", "Frankfurt", "Germany", "FRA", "EDDF", 50.033333, 8.570556, 364, 1, "E", "Europe/Berlin", "airport", "OurAirports", "Frankfurt Airport is Germany's busiest airport and one of the world's leading international hubs. Known for its massive size and extensive connections, FRA serves as the home base for Lufthansa and handles more international cargo than any other European airport. The airport is a crucial business travel hub and gateway to Central Europe."),

(15, "Zurich Airport", "Zurich", "Switzerland", "ZUR", "LSZH", 47.464722, 8.549167, 1416, 1, "E", "Europe/Zurich", "airport", "OurAirports", "Zurich Airport serves Switzerland's largest city and is known for its precision, efficiency, and stunning Alpine views. The airport reflects Swiss excellence in engineering and hospitality, offering premium services and convenient connections to ski resorts and mountain destinations. ZUR serves as a gateway to the Swiss Alps and European business centers."),

(16, "Hong Kong International Airport", "Hong Kong", "Hong Kong", "HKG", "VHHH", 22.308919, 113.914603, 28, 8, "U", "Asia/Hong_Kong", "airport", "OurAirports", "Hong Kong International Airport is built on an artificial island and serves as a major hub for Asia-Pacific travel. Known for its stunning architecture, efficient operations, and world-class shopping, HKG is consistently ranked among the world's best airports. The airport serves as a crucial link between mainland China and the rest of the world."),

(17, "Vancouver International Airport", "Vancouver", "Canada", "YVR", "CYVR", 49.193901, -123.184998, 4, -8, "A", "America/Vancouver", "airport", "OurAirports", "Vancouver International Airport serves the beautiful coastal city of Vancouver and is known for its stunning mountain and ocean views. The airport showcases Pacific Northwest culture with extensive art collections and local cuisine. YVR serves as a major gateway to Asia and is praised for its environmental initiatives and passenger experience."),

(18, "Mumbai Chhatrapati Shivaji Airport", "Mumbai", "India", "BOM", "VABB", 19.088686, 72.867919, 39, 5.5, "U", "Asia/Kolkata", "airport", "OurAirports", "Mumbai Chhatrapati Shivaji Maharaj International Airport is the busiest airport in India and serves the financial capital of the country. The airport handles millions of passengers connecting India to global destinations and serves as a major hub for domestic travel. Known for its vibrant atmosphere and cultural displays, BOM reflects the energy and diversity of Mumbai."),

(19, "São Paulo–Guarulhos International Airport", "São Paulo", "Brazil", "GRU", "SBGR", -23.435556, -46.473056, 2459, -3, "S", "America/Sao_Paulo", "airport", "OurAirports", "São Paulo–Guarulhos International Airport is Brazil's busiest airport and the main hub for South American air travel. Located in the metropolitan area of São Paulo, GRU serves as a crucial gateway between South America and the rest of the world. The airport handles both extensive domestic Brazilian traffic and international connections to Europe, North America, and Asia."),

(20, "Cairo International Airport", "Cairo", "Egypt", "CAI", "HECA", 30.121944, 31.405556, 382, 2, "U", "Africa/Cairo", "airport", "OurAirports", "Cairo International Airport serves the historic capital of Egypt and is one of Africa's busiest airports. Located northeast of Cairo, the airport serves as a gateway to ancient wonders including the Pyramids of Giza and the Sphinx. CAI connects Africa, the Middle East, and Europe, reflecting Egypt's historical role as a crossroads of civilizations."),

(21, "Johannesburg OR Tambo International Airport", "Johannesburg", "South Africa", "JNB", "FAJS", -26.139166, 28.246, 5558, 2, "U", "Africa/Johannesburg", "airport", "OurAirports", "OR Tambo International Airport is Africa's busiest airport and serves as the primary gateway to South Africa. Named after anti-apartheid activist Oliver Tambo, the airport connects Southern Africa to the world and serves as a major hub for exploring safari destinations, wine regions, and cultural attractions throughout the continent."),

(22, "Beijing Capital International Airport", "Beijing", "China", "PEK", "ZBAA", 40.080111, 116.584556, 116, 8, "U", "Asia/Shanghai", "airport", "OurAirports", "Beijing Capital International Airport serves China's capital city and is one of Asia's major aviation hubs. The airport, with its distinctive Terminal 3 designed by Norman Foster, serves as a gateway to Chinese culture, history, and the Great Wall of China. PEK handles enormous passenger volumes and reflects China's growing importance in global aviation."),

(23, "Mexico City International Airport", "Mexico City", "Mexico", "MEX", "MMMX", 19.436303, -99.072098, 7316, -6, "A", "America/Mexico_City", "airport", "OurAirports", "Mexico City International Airport serves the vibrant capital of Mexico and is one of Latin America's busiest airports. Located at high altitude in the Valley of Mexico, the airport serves as a major hub for travel throughout Latin America and connects Mexico to global destinations. MEX reflects Mexican culture and serves as a gateway to ancient civilizations and modern Mexico."),

(24, "Istanbul Airport", "Istanbul", "Turkey", "IST", "LTFM", 41.275278, 28.751944, 325, 3, "E", "Europe/Istanbul", "airport", "OurAirports", "Istanbul Airport is Turkey's main international airport and one of the world's largest airport terminals under one roof. Serving the transcontinental city of Istanbul, the airport connects Europe, Asia, Africa, and the Middle East. Known for its massive size, modern facilities, and strategic location, IST serves as Turkish Airlines' main hub and a major global transit point."),

(25, "Moscow Sheremetyevo International Airport", "Moscow", "Russia", "SVO", "UUEE", 55.972642, 37.414589, 622, 3, "U", "Europe/Moscow", "airport", "OurAirports", "Sheremetyevo International Airport is Russia's busiest airport and serves the capital city of Moscow. The airport serves as a major hub for Aeroflot and connects Russia to destinations worldwide. Known for its Soviet-era architecture mixed with modern terminals, SVO serves as a gateway to Russian culture, history, and the Trans-Siberian Railway."),

(26, "Warsaw Chopin Airport", "Warsaw", "Poland", "WAW", "EPWA", 52.165833, 20.967222, 362, 1, "E", "Europe/Warsaw", "airport", "OurAirports", "Warsaw Chopin Airport serves Poland's capital city and is Central Europe's major aviation hub. Named after the famous composer Frédéric Chopin, the airport serves as LOT Polish Airlines' main hub and connects Central Europe to global destinations. The airport serves as a gateway to Poland's rich cultural heritage and historic cities."),

(27, "Stockholm Arlanda Airport", "Stockholm", "Sweden", "ARN", "ESSA", 59.651944, 17.918611, 137, 1, "E", "Europe/Stockholm", "airport", "OurAirports", "Stockholm Arlanda Airport serves Sweden's capital and is Scandinavia's third-busiest airport. Known for its efficient design, environmental consciousness, and Scandinavian aesthetic, ARN serves as a gateway to Nordic culture and natural beauty. The airport connects Stockholm to global destinations and serves as a hub for exploring the Nordic countries."),

(28, "Copenhagen Airport", "Copenhagen", "Denmark", "CPH", "EKCH", 55.617917, 12.655972, 17, 1, "E", "Europe/Copenhagen", "airport", "OurAirports", "Copenhagen Airport serves Denmark's capital and is Scandinavia's busiest airport. Known for its award-winning design, sustainability initiatives, and Danish hygge atmosphere, CPH serves as SAS's main hub. The airport reflects Danish design principles and serves as a gateway to Scandinavian culture and the Baltic region."),

(29, "Helsinki-Vantaa Airport", "Helsinki", "Finland", "HEL", "EFHK", 60.317222, 24.963333, 179, 2, "E", "Europe/Helsinki", "airport", "OurAirports", "Helsinki-Vantaa Airport serves Finland's capital and is known for its innovative design and Finnish sauna facilities. The airport serves as Finnair's hub and is strategically positioned as a gateway between Europe and Asia. HEL reflects Finnish design excellence and serves as an introduction to Nordic culture and the Land of a Thousand Lakes."),

(30, "Oslo Airport Gardermoen", "Oslo", "Norway", "OSL", "ENGM", 60.193917, 11.100361, 681, 1, "E", "Europe/Oslo", "airport", "OurAirports", "Oslo Airport Gardermoen serves Norway's capital and is renowned for its environmental sustainability and Norwegian design. Built with extensive use of natural materials, the airport reflects Norway's commitment to environmental stewardship. OSL serves as a gateway to Norwegian fjords, Northern Lights, and midnight sun experiences.");

-- Add comprehensive descriptions for better RAG demonstrations
UPDATE airports SET description = CONCAT(description, ' The airport features modern facilities and serves as an important regional transportation hub.') WHERE description IS NOT NULL;

-- Verify the data insertion
SELECT COUNT(*) as airport_count FROM airports;
SELECT country, COUNT(*) as airports_per_country FROM airports GROUP BY country ORDER BY airports_per_country DESC;
