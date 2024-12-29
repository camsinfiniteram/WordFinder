import React, { useState } from 'react';
import { Button } from 'react-bootstrap';

function App() {
  const [word, setWord] = useState("");
  const [description, setDescription] = useState("");
  const [definition, setDefinition] = useState("");
  const [err, setErr] = useState(null);
  const [message, showMessage] = useState(false);
  const [loadingMessage] = useState("");

  const fetchWord = () => {
    fetch("http://localhost:5000/api/find_word", {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',  
      },
      body: JSON.stringify({ description }) 
    })
      .then(res => res.json())  
      .then(data => {
        if (data.error) {
          setErr(data.error);
          setWord("");
          setDefinition("");
        } else {
          setWord(data.word);
          setDefinition(data.definition);
          setErr(null);
        }
      })
      .catch(error => {
        setErr("Call it an egg, the way our server is scrambled! Please try again later.");
      });
  };

  const resetFields = () => {
    setWord("");
    setDescription("");
    setDefinition("");
  };

  const exit = () => {
    // when the button is clicked, the message will be displayed
    showMessage(true);
  }

  return (
    <div>
      <h1>What's That Word?</h1>
      <h4>Welcome, O Wordless One!</h4>
      <h4>Simply describe the word you're looking for, and we'll give you our best match.</h4>
        <input type="text" value={description}
        onChange={(e) => setDescription(e.target.value)}
        placeholder="Describe the word..." 
      />
      
      <Button onClick={fetchWord}>Submit</Button>

      {err && <p style={{ color: 'red' }}>{err}</p>}
      {
        word && (
          <div>
            <h3>Best Match:</h3>
            <p>{word}</p>
            <h3>Definition:</h3>
            <p>{definition}</p>
            <p>Would you like to search for another word?</p>
            <Button onClick={resetFields}>Yes</Button>
            <Button onClick={exit}>No</Button>
            
          </div>
        )}
        {message && <p>Thanks for stopping by! We hope you found what was on the tip of your tongue.</p>}   
    </div>
  );
}

export default App;
