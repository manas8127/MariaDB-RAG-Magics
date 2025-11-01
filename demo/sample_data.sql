-- Sample data for MariaDB RAG demo
-- Movie database with titles and descriptions for hackathon demonstration

-- Create database and verify Vector extension availability
CREATE DATABASE IF NOT EXISTS rag_demo;
USE rag_demo;

-- Verify MariaDB Vector extension is available
-- This will fail if Vector extension is not installed
SELECT 'MariaDB Vector extension check' as status, VEC_FromText('[0.1,0.2,0.3]') as test_vector;

-- Demo content table with proper indexes and constraints
DROP TABLE IF EXISTS demo_content;
CREATE TABLE demo_content (
    id INT PRIMARY KEY AUTO_INCREMENT,
    title VARCHAR(255) NOT NULL,
    content TEXT NOT NULL,
    genre VARCHAR(100) DEFAULT NULL,
    year INT DEFAULT NULL,
    content_vector VECTOR(384) DEFAULT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    -- Constraints
    CONSTRAINT chk_year CHECK (year IS NULL OR (year >= 1900 AND year <= 2030)),
    CONSTRAINT chk_title_length CHECK (CHAR_LENGTH(title) >= 1),
    CONSTRAINT chk_content_length CHECK (CHAR_LENGTH(content) >= 10)
) ENGINE=InnoDB;

-- Indexes for better performance
CREATE INDEX idx_genre ON demo_content(genre);
CREATE INDEX idx_year ON demo_content(year);
CREATE INDEX idx_title ON demo_content(title);
CREATE INDEX idx_created_at ON demo_content(created_at);

-- Sample movie data for demonstration (50+ records with variety of genres)
INSERT INTO demo_content (title, content, genre, year) VALUES 
-- Sci-Fi Movies
('Inception', 'A thief who steals corporate secrets through dream-sharing technology is given the inverse task of planting an idea into the mind of a C.E.O. Dom Cobb is a skilled thief, the absolute best in the dangerous art of extraction, stealing valuable secrets from deep within the subconscious during the dream state.', 'Sci-Fi', 2010),
('The Matrix', 'A computer programmer is led to fight an underground war against powerful computers who have constructed his entire reality with a system called the Matrix. Neo discovers that reality as he knows it is actually a computer simulation.', 'Sci-Fi', 1999),
('Interstellar', 'A team of explorers travel through a wormhole in space in an attempt to ensure humanity survival. Cooper, a former NASA pilot, must leave his family behind to lead an expedition beyond our galaxy to discover whether mankind has a future among the stars.', 'Sci-Fi', 2014),
('Blade Runner 2049', 'A young blade runner discovery of a long-buried secret leads him to track down former blade runner Rick Deckard. Officer K unearths a secret that could plunge what left of society into chaos.', 'Sci-Fi', 2017),
('Ex Machina', 'A young programmer is selected to participate in a ground-breaking experiment in synthetic intelligence by evaluating the human qualities of a highly advanced humanoid A.I. Caleb must determine if Ava truly has consciousness.', 'Sci-Fi', 2014),
('The Martian', 'An astronaut becomes stranded on Mars after his team assume him dead, and must rely on his ingenuity to find a way to signal to Earth that he is alive. Mark Watney must survive on a hostile planet using only his wit and limited supplies.', 'Sci-Fi', 2015),
('Arrival', 'A linguist works with the military to communicate with alien lifeforms after twelve mysterious spacecraft appear around the world. Dr. Louise Banks must decode their language before tensions lead to global war.', 'Sci-Fi', 2016),
('Gravity', 'Two astronauts work together to survive after an accident leaves them stranded in space. Dr. Ryan Stone and Matt Kowalski must find a way back to Earth after debris destroys their space shuttle.', 'Sci-Fi', 2013),
('Avatar', 'A paraplegic Marine dispatched to the moon Pandora on a unique mission becomes torn between following his orders and protecting the world he feels is his home. Jake Sully must choose between his human origins and the alien world he has come to love.', 'Sci-Fi', 2009),
('Star Wars: A New Hope', 'Luke Skywalker joins forces with a Jedi Knight, a cocky pilot, a Wookiee and two droids to save the galaxy from the Empire world-destroying battle station. A young farm boy discovers his destiny in an epic space opera.', 'Sci-Fi', 1977),

-- Action Movies
('Mad Max: Fury Road', 'In a post-apocalyptic wasteland, Max teams up with Furiosa to flee from cult leader Immortan Joe and his army in an armored war rig. A high-octane chase across the desert with spectacular practical effects and minimal dialogue.', 'Action', 2015),
('John Wick', 'An ex-hitman comes out of retirement to track down the gangsters that took everything from him. Keanu Reeves delivers intense action sequences in this stylish revenge thriller with incredible choreography.', 'Action', 2014),
('Die Hard', 'A New York police officer tries to save his wife and several others taken hostage by German terrorists during a Christmas party at the Nakatomi Plaza in Los Angeles. The ultimate action movie set in a single location.', 'Action', 1988),
('Terminator 2: Judgment Day', 'A cyborg, identical to the one who failed to kill Sarah Connor, must now protect her teenage son John Connor from a more advanced and powerful cyborg. Groundbreaking special effects in this sci-fi action masterpiece.', 'Action', 1991),
('The Dark Knight', 'Batman faces the Joker, a criminal mastermind who wants to plunge Gotham City into anarchy. Heath Ledger iconic performance elevates this superhero film to new heights of psychological complexity.', 'Action', 2008),

-- Drama Movies
('The Shawshank Redemption', 'Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency. A story of hope, friendship, and the human spirit triumph over adversity.', 'Drama', 1994),
('Forrest Gump', 'The presidencies of Kennedy and Johnson, Vietnam, Watergate, and other history unfold through the perspective of an Alabama man with an IQ of 75. Tom Hanks delivers a heartwarming performance in this American epic.', 'Drama', 1994),
('Goodfellas', 'The story of Henry Hill and his life in the mob, covering his relationship with his wife Karen Hill and his mob partners. Martin Scorsese masterful crime drama spans three decades of organized crime.', 'Drama', 1990),
('Pulp Fiction', 'The lives of two mob hitmen, a boxer, a gangster and his wife intertwine in four tales of violence and redemption. Quentin Tarantino nonlinear narrative revolutionized modern cinema.', 'Drama', 1994),
('The Godfather', 'The aging patriarch of an organized crime dynasty transfers control of his clandestine empire to his reluctant son. Francis Ford Coppola epic tale of power, family, and the American Dream.', 'Drama', 1972),

-- Comedy Movies
('Groundhog Day', 'A weatherman finds himself living the same day over and over again. Bill Murray delivers perfect comedic timing in this philosophical comedy about second chances and personal growth.', 'Comedy', 1993),
('The Grand Budapest Hotel', 'The adventures of Gustave H, a legendary concierge at a famous European hotel, and Zero Moustafa, the protégé who becomes his most trusted friend. Wes Anderson whimsical visual style creates a unique comedic experience.', 'Comedy', 2014),
('Superbad', 'Two co-dependent high school seniors are forced to deal with separation anxiety after their plan to stage a booze-soaked party goes awry. A raunchy coming-of-age comedy with heart.', 'Comedy', 2007),
('Anchorman', 'Ron Burgundy is San Diego top-rated newsman in the male-dominated broadcasting of the 1970s, but that about to change when ambitious reporter Veronica Corningstone arrives. Will Ferrell at his comedic best.', 'Comedy', 2004),
('The Hangover', 'Three buddies wake up from a bachelor party in Las Vegas, with no memory of the previous night and the bachelor missing. A wild comedy mystery that spawned multiple sequels.', 'Comedy', 2009),

-- Horror Movies
('Get Out', 'A young African-American visits his white girlfriend family estate, where he learns that many of its black visitors have gone missing. Jordan Peele masterful social thriller disguised as horror.', 'Horror', 2017),
('A Quiet Place', 'A family lives in silence while hiding from creatures that hunt by sound. John Krasinski creates tension through minimal dialogue and maximum suspense in this innovative horror film.', 'Horror', 2018),
('Hereditary', 'A grieving family is haunted by tragedy and disturbing secrets. Ari Aster psychological horror explores family trauma with genuinely terrifying imagery and atmosphere.', 'Horror', 2018),
('The Conjuring', 'Paranormal investigators Ed and Lorraine Warren work to help a family terrorized by a dark presence in their farmhouse. Classic supernatural horror with excellent character development.', 'Horror', 2013),
('Halloween', 'Fifteen years after murdering his sister on Halloween night 1963, Michael Myers escapes from a mental hospital and returns to the small town of Haddonfield to kill again. The film that defined the slasher genre.', 'Horror', 1978),

-- Romance Movies
('The Princess Bride', 'A bedridden boy grandfather reads him the story of a farmboy-turned-pirate who encounters numerous obstacles, enemies and allies in his quest to be reunited with his true love. A perfect blend of romance, adventure, and comedy.', 'Romance', 1987),
('Titanic', 'A seventeen-year-old aristocrat falls in love with a kind but poor artist aboard the luxurious, ill-fated R.M.S. Titanic. James Cameron epic romance set against historical tragedy.', 'Romance', 1997),
('Casablanca', 'A cynical American expatriate struggles to decide whether or not he should help his former lover and her fugitive husband escape French Morocco. The ultimate classic romance set during World War II.', 'Romance', 1942),
('When Harry Met Sally', 'Harry and Sally have known each other for years, and are very good friends, but they fear sex would ruin the friendship. Rob Reiner exploration of friendship and love with memorable dialogue.', 'Romance', 1989),
('Eternal Sunshine of the Spotless Mind', 'When their relationship turns sour, a couple undergoes a medical procedure to have each other erased from their memories. Charlie Kaufman innovative script explores love and memory.', 'Romance', 2004),

-- Thriller Movies
('Se7en', 'Two detectives, a rookie and a veteran, hunt a serial killer who uses the seven deadly sins as his motives. David Fincher dark thriller with an unforgettable ending.', 'Thriller', 1995),
('Silence of the Lambs', 'A young F.B.I. cadet must receive the help of an incarcerated and manipulative cannibal killer to help catch another serial killer. Anthony Hopkins chilling performance as Hannibal Lecter.', 'Thriller', 1991),
('Gone Girl', 'With his wife disappearance having become the focus of an intense media circus, a man sees the spotlight turned on him when it suspected that he may not be innocent. Gillian Flynn psychological thriller with unreliable narrators.', 'Thriller', 2014),
('Zodiac', 'In the late 1960s/early 1970s, a San Francisco cartoonist becomes an amateur detective obsessed with tracking down the Zodiac Killer. David Fincher meticulous procedural thriller based on true events.', 'Thriller', 2007),
('No Country for Old Men', 'Violence and mayhem ensue after a hunter stumbles upon a drug deal gone wrong and more than two million dollars in cash near the Rio Grande. The Coen Brothers neo-western thriller with philosophical depth.', 'Thriller', 2007),

-- Fantasy Movies
('The Lord of the Rings: The Fellowship of the Ring', 'A meek Hobbit from the Shire and eight companions set out on a journey to destroy the powerful One Ring and save Middle-earth from the Dark Lord Sauron. Peter Jackson epic adaptation of Tolkien masterpiece.', 'Fantasy', 2001),
('Pan Labyrinth', 'In the Falangist Spain of 1944, the bookish young stepdaughter of a sadistic army officer escapes into an eerie but captivating fantasy world. Guillermo del Toro dark fairy tale for adults.', 'Fantasy', 2006),
('The Shape of Water', 'At a top secret research facility in the 1960s, a lonely janitor forms a unique relationship with an amphibious creature that is being held in captivity. Del Toro romantic fantasy about love transcending boundaries.', 'Fantasy', 2017),
('Big Fish', 'A frustrated son tries to determine the fact from fiction in his dying father life. Tim Burton whimsical tale about storytelling and the relationship between fathers and sons.', 'Fantasy', 2003),
('Edward Scissorhands', 'An artificial man, who was incompletely constructed and has scissors for hands, leads a solitary life. Then one day, a suburban lady meets him and introduces him to her world. Burton gothic fairy tale about acceptance and difference.', 'Fantasy', 1990),

-- Animation Movies
('Spirited Away', 'During her family move to the suburbs, a sullen 10-year-old girl wanders into a world ruled by gods, witches, and spirits, and where humans are changed into beasts. Hayao Miyazaki masterpiece of imagination.', 'Animation', 2001),
('Toy Story', 'A cowboy doll is profoundly threatened and jealous when a new spaceman figure supplants him as top toy in a boy room. Pixar groundbreaking computer-animated film that revolutionized animation.', 'Animation', 1995),
('WALL-E', 'In the distant future, a small waste-collecting robot inadvertently embarks on a space journey that will ultimately decide the fate of mankind. Pixar environmental message wrapped in a love story.', 'Animation', 2008),
('Inside Out', 'After young Riley is uprooted from her Midwest life and moved to San Francisco, her emotions - Joy, Fear, Anger, Disgust and Sadness - conflict on how best to navigate a new city, house, and school. Pixar exploration of emotional intelligence.', 'Animation', 2015),
('Your Name', 'Two strangers find themselves linked in a bizarre way. When a connection forms, will distance be the only thing to keep them apart? Makoto Shinkai beautiful anime about connection across time and space.', 'Animation', 2016),

-- Western Movies
('The Good, the Bad and the Ugly', 'A bounty hunting scam joins two men in an uneasy alliance against a third in a race to find a fortune in gold buried in a remote cemetery. Sergio Leone spaghetti western masterpiece.', 'Western', 1966),
('Unforgiven', 'Retired Old West gunslinger William Munny reluctantly takes on one last job, with the help of his old partner Ned Logan and a young man, The "Schofield Kid." Clint Eastwood deconstruction of the western genre.', 'Western', 1992),
('True Grit', 'A stubborn teenager enlists the help of a tough U.S. Marshal to track down her father murderer. The Coen Brothers remake of the John Wayne classic with Hailee Steinfeld breakout performance.', 'Western', 2010),
('Butch Cassidy and the Sundance Kid', 'Wyoming, early 1900s. Butch Cassidy and The Sundance Kid are the leaders of a band of outlaws. After a train robbery goes wrong they find themselves on the run with a posse hard on their heels. Paul Newman and Robert Redford iconic buddy western.', 'Western', 1969),
('Django Unchained', 'With the help of a German bounty hunter, a freed slave sets out to rescue his wife from a brutal Mississippi plantation owner. Quentin Tarantino revisionist western with social commentary.', 'Western', 2012);