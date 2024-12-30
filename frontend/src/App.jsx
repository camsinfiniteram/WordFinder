import React, { useState } from 'react';
import { Button } from 'react-bootstrap';
import './App.css';

/**
 * Main functionalities:
 * - Fetches word from server based on user-inputted description
 * - Displays word and its definition
 * - Allows user to search for another word or exit app
 * - Displays error message if server is down
 * - Resets fields when user searches for another word
 */

function App() {
  const [word, setWord] = useState("");
  const [description, setDescription] = useState("");
  const [definition, setDefinition] = useState("");
  const [err, setErr] = useState(null);
  const [message, showMessage] = useState(false);
  const [loadingMessage, setLoadingMessage] = useState("");

  const fetchWord = () => {
    setLoadingMessage("Searching for the word...");
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
    setLoadingMessage("");
    showMessage("");
  };

  const exit = () => {
    showMessage(true);
  }

  return (
    <div>
      <h1>What's That Word?</h1>
      <h3>Welcome, O Wordless One!</h3>
      <h4>Simply describe the word you're looking for, and we'll give you our best match.</h4>
        <input type="text" value={description}
        onChange={(e) => setDescription(e.target.value)}
        placeholder="Describe the word..." 
      />
      
      <Button onClick={fetchWord}>Submit</Button>
      {loadingMessage && <p>{loadingMessage}</p>}

      {err && <p style={{ color: 'red' }}>{err}</p>}
      {
        word && (
          <div>
            <h3>Best Match:</h3>
            <p><b>{word}</b></p>
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
