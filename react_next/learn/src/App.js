import react from "react"

function Header(props) {
  // let title = props.title
  // return <h1>{title ? title : "Default Title"}</h1>
  return <h1>{props.title}</h1>
}

function HomePage() {
  const title = "Develop. Preview. Ship."
  const names = ['Ada Lovelace', 'Grace Hopper', 'Margaret Hamilton']

  const [likes, setLikes] = react.useState(0)
  function handleClick() {
    console.log("Increment by 1")
    setLikes(likes + 1)
  }

  return (
    <div>
      <Header title={title} />
      <ul>
        {names.map(name => (
          <li>{name}</li>
        ))}
      </ul>
      <button onClick={handleClick}>Like {(likes)}</button>
    </div>
  )
}

function App() { return <div><HomePage /></div>}

export default App